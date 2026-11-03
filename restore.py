import os
import sys
import json
import glob
import subprocess
import time
from pathlib import Path

# =========================================================================
# 0. 基础依赖与安全加载（防止未安装 PyTorch 时导致整个脚本无法启动）
# =========================================================================
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_FRAMES = 1000  # 每 1000 帧存为一个独立安全切片（约 40 秒），断电不丢进度

# =========================================================================
# 1. 原生 PyTorch Real-ESRGAN 网络架构
# =========================================================================
_BaseModule = nn.Module if HAS_TORCH else object

class ResidualDenseBlock_5C(_BaseModule):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(_BaseModule):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

class FlexibleRRDBNet(_BaseModule):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(FlexibleRRDBNet, self).__init__()
        self.scale = scale
        self.in_nc = in_nc
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if scale == 4:
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.in_nc == 12:
            x = F.pixel_unshuffle(x, downscale_factor=2)
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out

def build_esrgan_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
    in_nc = state_dict['conv_first.weight'].shape[1]
    scale = 4 if 'conv_up2.weight' in state_dict else 2
    print(f"--> AI 权重加载: scale={scale}x, 输入通道={in_nc}")

    model = FlexibleRRDBNet(in_nc=in_nc, out_nc=3, nf=64, nb=23, gc=32, scale=scale)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    if device.type == 'cuda':
        model = model.half()
    return model

# =========================================================================
# 2. 电影色彩校正、时长检测与工具函数
# =========================================================================
def get_video_duration(video_path: Path) -> float:
    """通过 ffprobe 高速获取视频精确时长（秒）"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path.resolve())
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass

    # 备用方案：通过 ffmpeg 头部信息正则解析
    try:
        cmd_fb = ["ffmpeg", "-i", str(video_path.resolve())]
        res_fb = subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d\.]+)", res_fb.stderr)
        if m:
            h, mins, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mins * 60 + s
    except Exception:
        pass
    return 0.0

def get_movie_color_fix_filter(remove_yellow: bool = True) -> str:
    """老电影去发黄、去暗沉、白平衡自适应校准滤镜链"""
    if not remove_yellow:
        return "null"
    return "colorbalance=rs=-0.04:gs=-0.02:bs=0.06:rm=-0.03:gm=-0.01:bm=0.05,eq=contrast=1.06:brightness=0.01:saturation=1.03,curves=all='0/0 0.05/0.03 1/1'"

def calculate_adaptive_target_size(orig_w: int, orig_h: int, target_bound: int = 1920):
    aspect = orig_w / orig_h
    if aspect >= 1.0:
        tw = target_bound
        th = int(tw / aspect)
    else:
        th = target_bound
        tw = int(th * aspect)
    tw = tw if tw % 2 == 0 else tw + 1
    th = th if th % 2 == 0 else th + 1
    return tw, th

def verify_chunk_validity(chunk_path: Path, expected_frames: int) -> bool:
    if not chunk_path.exists() or chunk_path.stat().st_size < 1024 or cv2 is None:
        return False
    cap = cv2.VideoCapture(str(chunk_path.resolve()))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return abs(count - expected_frames) <= 2

# =========================================================================
# 3. 核心功能 1：AI 深度超分 + 断点保存 + 去黄
# =========================================================================
def process_movie_ai_upscale_with_resume(
    esrgan_model, device, input_path: Path, output_path: Path, 
    target_bound: int = 1920, remove_yellow: bool = True, max_frames: int = 0
):
    if cv2 is None or not HAS_TORCH:
        print("❌ 未安装 opencv-python 或 torch，无法执行 AI 超分！")
        return

    cap = cv2.VideoCapture(str(input_path.resolve()))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w, target_h = calculate_adaptive_target_size(orig_w, orig_h, target_bound)

    print(f"\n🎬 原始电影规格: {orig_w}x{orig_h} (原生画幅: {orig_w/orig_h:.2f}:1, 帧率: {fps:.2f}fps)")
    print(f"✨ 目标重构规格: {target_w}x{target_h} (总帧数: {total_frames} 帧)")

    chunk_dir = input_path.parent / f"temp_resume_{input_path.stem}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    progress_file = chunk_dir / "progress.json"

    total_chunks = (total_frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    completed_chunks = set()

    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            completed_chunks = set(data.get("completed", []))
            print(f"🔄 检测到历史进度！已完成 {len(completed_chunks)}/{total_chunks} 个分段。")
        except Exception:
            completed_chunks = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for chunk_idx in range(total_chunks):
        start_frame = chunk_idx * CHUNK_FRAMES
        end_frame = min(start_frame + CHUNK_FRAMES, total_frames)
        expected_len = end_frame - start_frame
        chunk_file = chunk_dir / f"chunk_{chunk_idx:05d}.mp4"

        if chunk_idx in completed_chunks and verify_chunk_validity(chunk_file, expected_len):
            print(f"⏩ [切片跳过 {chunk_idx+1:04d}/{total_chunks:04d}] 帧区间: {start_frame}~{end_frame} (已完成)")
            continue

        print(f"\n🚀 [开始推理切片 {chunk_idx+1:04d}/{total_chunks:04d}] 帧区间: {start_frame}~{end_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(str(chunk_file.resolve()), fourcc, fps, (target_w, target_h))

        with torch.no_grad():
            for f_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                img = frame[:, :, ::-1] / 255.0
                img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
                if device.type == 'cuda':
                    img = img.half()

                output = esrgan_model(img)
                output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy().transpose((1, 2, 0))
                enhanced = (output[:, :, ::-1] * 255.0).round().astype('uint8')

                if (enhanced.shape[1], enhanced.shape[0]) != (target_w, target_h):
                    final_frame = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    final_frame = enhanced

                writer.write(final_frame)
                if (f_idx + 1) % 30 == 0 or (f_idx + 1) == end_frame:
                    overall_pct = ((f_idx + 1) / total_frames) * 100
                    print(f"\r  ⚡ AI 处理进度: {f_idx+1}/{total_frames} 帧 ({overall_pct:.2f}%)", end="", flush=True)

        writer.release()
        if verify_chunk_validity(chunk_file, expected_len):
            completed_chunks.add(chunk_idx)
            progress_file.write_text(json.dumps({"completed": sorted(list(completed_chunks))}), encoding="utf-8")

    cap.release()
    print()

    print("📦 正在合并切片，并应用【老电影胶片去黄与白平衡调校】...")
    concat_list_file = chunk_dir / "concat_list.txt"
    chunk_lines = [f"file '{(chunk_dir / f'chunk_{c:05d}.mp4').resolve().as_posix()}'\n" for c in range(total_chunks)]
    concat_list_file.write_text("".join(chunk_lines), encoding="utf-8")

    color_filter = get_movie_color_fix_filter(remove_yellow)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list_file.resolve()),
        "-i", str(input_path.resolve()),
        "-vf", color_filter,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium",
        "-c:a", "copy",
        str(output_path.resolve())
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

    if res.returncode == 0 and output_path.exists():
        print(f"🎉 影片修复完毕！已生成: {output_path.name}")
        for c in range(total_chunks):
            p = chunk_dir / f"chunk_{c:05d}.mp4"
            if p.exists():
                p.unlink()
        if concat_list_file.exists():
            concat_list_file.unlink()
        if progress_file.exists():
            progress_file.unlink()
        try:
            chunk_dir.rmdir()
        except Exception:
            pass

# =========================================================================
# 4. 核心功能 2：胶片级极速轻量调校 + 去黄 (带实时百分比与速度进度)
# =========================================================================
def process_movie_lightweight(input_path: Path, output_path: Path, target_bound: int = 1920, remove_yellow: bool = True, max_sec: int = 0):
    total_dur = get_video_duration(input_path)
    if max_sec > 0:
        total_dur = min(total_dur, float(max_sec)) if total_dur > 0 else float(max_sec)

    print(f"\n🎬 正在快速优化影片: {input_path.name}")
    if total_dur > 0:
        print(f"⏱️ 视频时长: {total_dur:.1f} 秒 ({total_dur/60:.1f} 分钟)")

    color_filter = get_movie_color_fix_filter(remove_yellow)
    vf_list = [
        "hqdn3d=2.0:1.5:3.0:2.5",  # 极速且通透的 3D 胶片降噪（耗时仅为 nlmeans 的 1/30）
        "cas=0.35",                # 自适应锐化边缘
        color_filter,              # 去黄与白平衡校准
        f"scale='if(gt(a,1),{target_bound},-2)':'if(gt(a,1),-2,{target_bound})':flags=lanczos"
    ]
    full_vf = ",".join([f for f in vf_list if f != "null"])

    cmd = ["ffmpeg", "-y"]
    if max_sec > 0:
        cmd.extend(["-t", str(max_sec)])
    cmd.extend([
        "-i", str(input_path.resolve()),
        "-vf", full_vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        "-progress", "pipe:1",
        str(output_path.resolve())
    ])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")

    out_time_sec = 0.0
    fps_val = "0"
    speed_val = "0x"

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                out_time_sec = ms / 1_000_000.0
            except ValueError:
                pass
        elif line.startswith("fps="):
            fps_val = line.split("=")[1].strip()
        elif line.startswith("speed="):
            speed_val = line.split("=")[1].strip()

        if total_dur > 0 and out_time_sec > 0:
            pct = min(100.0, (out_time_sec / total_dur) * 100)
            print(f"\r  ⚡ 调校进度: {pct:.1f}% | 已完成: {out_time_sec:.1f}s/{total_dur:.1f}s | 帧率: {fps_val} fps | 速度: {speed_val}", end="", flush=True)

    print()
    if proc.returncode == 0 and output_path.exists():
        print(f"\n🎉 调校大功告成！已成功输出: {output_path.name}")
        return True
    else:
        print("\n❌ 视频处理异常。")
        return False

# =========================================================================
# 5. 核心功能 3：前 30 秒快速 PK 对比测试
# =========================================================================
def run_30s_pk_test(movie: Path, target_bound: int = 1920):
    print("\n" + "=" * 65)
    print(f"🧪 正在对影片【{movie.name}】执行前 30 秒快速 PK 测试...")
    print("=" * 65)

    sample_raw = SCRIPT_DIR / "temp_sample_30s.mp4"
    cmd_cut = [
        "ffmpeg", "-y", "-ss", "0", "-t", "30",
        "-i", str(movie.resolve()),
        "-c", "copy", str(sample_raw.resolve())
    ]
    subprocess.run(cmd_cut, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not sample_raw.exists() or sample_raw.stat().st_size < 1024:
        cmd_cut_reencode = [
            "ffmpeg", "-y", "-ss", "0", "-t", "30",
            "-i", str(movie.resolve()),
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy",
            str(sample_raw.resolve())
        ]
        subprocess.run(cmd_cut_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_mode2 = SCRIPT_DIR / f"{movie.stem}_PK_MODE2_LIGHT.mp4"
    print("\n▶️ [测试 1/2] 正在运行 选项 2 (轻量降噪+CAS锐化+去黄)...")
    t0 = time.time()
    process_movie_lightweight(sample_raw, out_mode2, target_bound=target_bound, remove_yellow=True)
    mode2_time = time.time() - t0
    print(f"✅ 选项 2 处理完成！耗时: {mode2_time:.1f} 秒 -> {out_mode2.name}")

    out_mode1 = SCRIPT_DIR / f"{movie.stem}_PK_MODE1_AI.mp4"
    print("\n▶️ [测试 2/2] 正在运行 选项 1 (Real-ESRGAN AI 深度超分+去黄)...")
    weights = list(SCRIPT_DIR.glob("*ESRGAN*.pth"))
    if not HAS_TORCH:
        print("⚠️ 未安装 PyTorch，跳过模式 1 测试。")
        mode1_time = 0
    elif not weights:
        print("⚠️ 未在目录下找到 Real-ESRGAN 模型权重，跳过 AI 超分测试。")
        mode1_time = 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_esrgan_model(weights[0], device)
        t0 = time.time()
        process_movie_ai_upscale_with_resume(model, device, sample_raw, out_mode1, target_bound=target_bound, remove_yellow=True)
        mode1_time = time.time() - t0
        print(f"✅ 选项 1 处理完成！耗时: {mode1_time:.1f} 秒 -> {out_mode1.name}")

    if sample_raw.exists():
        sample_raw.unlink()

    print("\n" + "=" * 65)
    print("📊 【前 30 秒测试诊断报告】")
    print(f"  ⚡ 选项 2 (轻量纯滤镜) : 耗时 {mode2_time:.1f} 秒 (全片 90 分钟预计约 {mode2_time*180/60:.1f} 分钟)")
    if mode1_time > 0:
        print(f"  🤖 选项 1 (AI 深度重构): 耗时 {mode1_time:.1f} 秒 (全片 90 分钟预计约 {mode1_time*180/60:.1f} 分钟)")
    print("-" * 65)
    print("👉 可在播放器中并排打开以下生成的文件，直观对比画质与通透度：")
    print(f"   1. {out_mode2.name}")
    if mode1_time > 0:
        print(f"   2. {out_mode1.name}")
    print("=" * 65 + "\n")

# =========================================================================
# 6. 主菜单交互
# =========================================================================
def scan_movies():
    exts = {".mp4", ".mkv", ".mov", ".avi", ".ts"}
    return [
        f for f in SCRIPT_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in exts
        and not f.name.startswith("temp_")
        and not "_PK_" in f.name
        and not f.name.endswith("_enhanced.mp4")
    ]

def main():
    print("=" * 65)
    print("      🎞️ 老旧电影与长视频重塑大师 (带去黄与断电续跑)")
    print("=" * 65)
    print(" 1. 🤖 [整片 AI 深度超分 + 老片去黄] (需 GPU 与模型，画质质变，支持断电续跑)")
    print(" 2. ⚡ [整片 胶片降噪 + CAS锐化 + 去黄] (无需模型/显卡，实时进度，速度极快)")
    print(" 3. 🧪 [前 30 秒画质效果 PK 测试] (截取开头 30 秒分别跑两套方案对比)")
    print(" 0. 退出")
    print("=" * 65)

    choice = input("请选择模式 (0-3): ").strip()
    if choice not in ["1", "2", "3"]:
        return

    movies = scan_movies()
    if not movies:
        print(f"⚠️ 当前目录 {SCRIPT_DIR} 下没有找到待处理视频！")
        return

    print(f"\n📂 待处理影片列表:")
    for i, m in enumerate(movies, 1):
        print(f"  [{i}] {m.name}")

    sel = input(f"请输入编号 (1-{len(movies)}, 或输入 all 处理全部): ").strip().lower()
    targets = movies if sel == "all" else [movies[int(sel) - 1]] if sel.isdigit() and 1 <= int(sel) <= len(movies) else []
    if not targets:
        return

    if choice == "3":
        run_30s_pk_test(targets[0], target_bound=1920)
        return

    res_input = input("长边基准分辨率 [默认 1920 (1080P)]: ").strip()
    target_bound = int(res_input) if res_input.isdigit() else 1920

    if choice == "1":
        if not HAS_TORCH:
            print("\n❌ 模式 1 需要安装 PyTorch！请在终端运行：")
            print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n")
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = list(SCRIPT_DIR.glob("*ESRGAN*.pth"))
        if not weights:
            print(f"❌ 当前目录下未找到 Real-ESRGAN 模型权重文件 (*.pth)！")
            return
        model = build_esrgan_model(weights[0], device)

        for m in targets:
            out_file = SCRIPT_DIR / f"{m.stem}_enhanced.mp4"
            process_movie_ai_upscale_with_resume(model, device, m, out_file, target_bound, remove_yellow=True)

    elif choice == "2":
        for m in targets:
            out_file = SCRIPT_DIR / f"{m.stem}_enhanced.mp4"
            process_movie_lightweight(m, out_file, target_bound, remove_yellow=True)

if __name__ == "__main__":
    main()