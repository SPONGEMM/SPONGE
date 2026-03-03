SPONGE 内 pixi 环境名及其作用为：

- default
  完全使用系统依赖（default环境仍需显式指定-e default）

- cpu
  由pixi管理依赖的无SIMD优化的CPU版本

- cuda126
  由pixi管理依赖的cuda12.6 GPU版本

- cuda128
  由pixi管理依赖的cuda12.8 GPU版本

- cuda130
  由pixi管理依赖的cuda13.0 GPU版本

- dev-xxx
  上述的xxx的开发者模式版本

- ref-gen
  生成对照参考的开发者环境
