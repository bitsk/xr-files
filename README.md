# 空间文件 (xr-files)

openKylin XR 的原生三维空间文件管理器，用于替代传统 2D 平面文件管理器（如 Peony）。基于 **Rust + wgpu (Vulkan / WGSL) + 跨架构硬件 SIMD** 构建，**默认作为空间桌面窗口与 HyperSpace (`xr-shell` + `xr-window-server`) 共存**——保留晴空全景背景、三岛式空间 Dock 栏与其他应用窗口，并直接复用 HyperSpace 统一的空间拖动、双柄/边角缩放、柱面弯折（`curve`）、最小化与关闭交互；同时保留显式 `--xr` 独立 OpenXR 双目会话与 `--headless` 离屏校验模式。

## 核心架构与空间特性

1. **透明空间挂件与六卡货架**
   - 默认 1600×900 预乘透明 Wayland 画布，六枚 72mm 导航球、独立路径/搜索胶囊、3×2 张 360×265mm 磁贴和弹簧翻页。主标题 33mm（设计距离 1.24m 处 26.6 dmm），独立热区至少 68mm。
   - 256² 图标与卡片缩略图、512² Quick Look 纹理、最高 96px 字形光栅及 SIMD mipmap。没有整窗底板、Hero 横幅或常驻右栏；文件夹悬停异步展开最多三张扇形子项预览。
   - 3D 体积星图保留为按需切换视图。HyperSpace 为本应用取消矩形外沿和阴影，继续提供统一拖拽、缩放、弯折、最小化与关闭。

2. **跨平台硬件 SIMD 加速 (`src/simd.rs`)**
   - 自动按目标架构启用 `x86_64` **AVX2 / SSE2**、`aarch64` **NEON**，并提供 `riscv64` 精确标量回退：
     - **SoA 三维射线包围盒批量求交 (`SoaAabbBatch`)**：8 路/4 路并行求交，内置 `3.2°` 远场动态热区容差，并严格遵循 `AUDIT-01` 同平面真实几何命中绝对优先于扩展热区。
     - **SIMD 大小写折叠子串与边界模糊匹配**：极速过滤与排序海量目录条目。
     - **SIMD 图像 2×2 Box 降采样与预乘 Alpha 转换**：在 CPU 侧无损生成缩略图与 CJK 字形图集的完整 Mipmap 链，配合 GPU 线性三线性与各向异性采样消除转头高频闪烁。
     - **SIMD 字节流扫描与香农信息熵**：并行统计换行符、NUL 二进制特征与 4 路展开直方图熵值。

3. **完整文件管理闭环与 6DoF 双手交互**
   - 支持新建文件夹、重命名、多选、复制/剪切/粘贴（自动防覆盖命名）、符合 FreeDesktop Trash 1.0 标准的回收站移入与原位还原，以及全操作可逆撤销栈（`Undo`）。
   - 双手射线、施密特阈值（扳机 0.65/0.35、抓握 0.60/0.35）、120ms / 2.4° 按下目标锁、直接几何命中优先和 28mm 磁吸浮起。
   - A/X（桌面右键）按住呼出六扇区轮盘，手腕偏转后松开即执行；中心短按驻留后可逐项点击。扳机/捏合静止长按 340ms 使用同一轮盘，中心与 B/Y / Esc 安全取消；排序扇区持续按住 700ms 切换隐藏项。
   - 文件卡片拖入左下暂存囊，可跨多个目录积累路径引用；切换目标目录后拖出暂存囊或使用投放操作，一次移动并可整体撤销。清空暂存仅移除引用。
   - 检视徽标、径向轮盘、Space 或以文件路径启动均打开中央 Quick Look。图片/视频海报、十行文本/二进制摘要与 OBJ/STL/GLB 全息模型按需展示；按住舞台拖动可连续旋转模型。Space、B/Y、Esc、关闭球或空白遮罩返回货架。
   - 空白横拖 90mm、摇杆/滚轮和页码左右半区均可翻页。保留 Ctrl+C/X/V/Z/A、F2、Delete 等键盘文件操作。


## 构建、测试与打包

构建无需外部 `glslangValidator` 或 Cairo/Pango 头依赖（纯 Rust WGSL + `ab_glyph` 字形光栅）：

```sh
CARGO_TARGET_DIR=/tmp/xr-files-tests cargo test --locked
```

正式构建和 Debian 打包由 `xr-image-tools` 从已提交的组件锁导出源码，输出目录必须位于源码树外：

```sh
python3 ../xr-image-tools/tools/components.py build --profile files --output /tmp/xr-files-build --only xr-files
```

测试包含径向拨盘/长按取消、防抖锁、跨目录暂存投放与撤销、异步目录透视、模型手腕旋转、热区/字号、真实离屏双目视差，以及 Argb8888 四角透明和全帧预乘不变量。可设置 `XR_FILES_TEST_CAPTURE_DIR` 导出 GPU 测试图；这些离屏结果不代替实机扫描输出或佩戴验收。

## 运行与智能控制接口 v1

```sh
xr-files                          # 默认与 HyperSpace 空间桌面 (xr-wayland) 或桌面窗口共存
xr-files --window ~/Documents     # 强制以空间桌面共存面板模式运行 (GPU 渲染 -> wl_shm Argb8888 预乘透明合成)
xr-files --xr ~/Documents         # 显式以独立前景 OpenXR 双目立体会话运行 (含 BLEND_TEXTURE_SOURCE_ALPHA)
xr-files --treemap ~              # 启动时直接进入 3D 体积星图视图
xr-files --headless --capture-dir /tmp/xr-files-cap
```

运行期间提供本地控制 socket `$XDG_RUNTIME_DIR/xr-files/control.sock`（权限 `0600`）：

```sh
xr-files ctl '{"op":"status"}'
xr-files ctl '{"op":"search","query":"openkylin"}'
xr-files ctl '{"op":"set_view","view":"treemap3d"}'
xr-files ctl '{"op":"mkdir","name":"空间工作区"}'
xr-files ctl '{"op":"undo"}'
xr-files ctl '{"op":"quit"}'
```
