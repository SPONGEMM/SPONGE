---
name: sponge-skill-installer
description: >
  询问为 SPONGE 安装、链接、修复或检查 skills 时使用。
  适用于 AI 助手的本地或全局 skills 配置。
---

# SPONGE Skill Installer

本技能用于把 SPONGE 仓库自带的 `skills/` 正确接入 AI 助手的 skills 目录。

## 先判断安装目标

先根据用户意图判断是安装至项目目录还是全局目录：

- 如果用户要在当前 SPONGE 仓库里开发、调试、补 benchmark、改插件或持续协作，使用项目级目录。
  通常开发代码是在项目进行的，使用此项。
- 如果用户只是希望助手未来在任意目录都能使用SPONGE进而调用SPONGE skill，使用全局目录。
  通常使用SPONGE进行分子模拟是在全局使用的，使用此项。

## 安装 skills

将需要的技能链接或复制至对应的AI助手目录，其中场景的AI助手的全局和项目目录包括：

- claude: 全局 `~/.claude/skills`；项目 `./.claude/skills`
- codex: 全局 `~/.agents/skills`；项目 `./.agents/skills`
- gemini: 全局 `~/.gemini/skills`；项目 `./.gemini/skills`
- cursor：全局 `~/.cursor/skills`；项目 `./.cursor/skills`
- qwen: 全局 `~/.qwen/skills`；项目 `./.qwen/skills`

## 提供的 skills 内容

- sponge-skill-installer: 指导安装SPONGE skill的skill
- sponge-benchmark-creator: 指导创建SPONGE benchmark的skill
- sponge-prips-mlp: 指导使用 PRIPS 对接 MACE 等机器学习/神经网络势并在 SPONGE 中验证与 benchmark
