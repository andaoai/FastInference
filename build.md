# FastInference 项目 Release/Debug 双构建说明

本项目采用 Conan 管理依赖，CMake 进行构建。推荐分别为 Release 和 Debug 构建使用独立的 build 目录，互不干扰，方便开发和调试。

---

## 1. 目录结构建议

```
FastInference/
├── build-release/   # Release 构建目录
├── build-debug/     # Debug 构建目录
├── src/
├── CMakeLists.txt
├── conanfile.txt
└── ...
```

---

## 2. 安装依赖并生成 toolchain

### Release 依赖

```bash
conan install . -s build_type=Release --output-folder=build-release --build=missing
```

### Debug 依赖

```bash
conan install . -s build_type=Debug --output-folder=build-debug --build=missing
```

---

## 3. 配置 CMake 构建

### Release 构建

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build-release/conan_toolchain.cmake -S . -B build-release
```

### Debug 构建

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=build-debug/conan_toolchain.cmake -S . -B build-debug
```

---

## 4. 编译

### Release 编译

```bash
cmake --build build-release
```

### Debug 编译

```bash
cmake --build build-debug
```

---

## 5. 运行

- Release 版本可执行文件路径：`build-release/fast_inference`
- Debug 版本可执行文件路径：`build-debug/fast_inference`

---

## 6. 调试（以 Debug 版本为例）

```bash
gdb build-debug/fast_inference
```
或在 VS Code 里选择 build-debug 目录下的可执行文件进行调试。

---

## 7. 常见问题

- **切换构建类型时请勿混用 build 目录。**
- 每次更换依赖或 CMake 配置，建议重新 conan install 和 cmake。
- VS Code 用户可通过 CMake Tools 插件配置多套构建目录，左下角一键切换。

---

## 8. VS Code 用户建议

- 在 CMake 工具栏中添加 `build-release` 和 `build-debug` 两套配置，分别指向不同的 build 目录和 toolchain 文件。
- 切换配置后点击"全部生成"即可。

---

如需更详细的 VS Code 配置（如 CMakePresets.json 或 launch.json 示例），请告知！ 