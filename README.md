# BestChoice

MACD 选股台本地工具，包含当前浅色版界面和从历史会话恢复的早期版本。

## 启动

```bash
./start.command
```

默认打开当前版本：

```text
http://localhost:8765/
```

## 深色第一版复现

```bash
./start_dark.command
```

打开：

```text
http://localhost:8765/dark
```

深色复现版固定使用第一版规则：

- `EMA(10,22,8)`
- 持股 `15` 天
- 额比阈值 `>= 1.42`
- 价格位置 `<= 60%`
- 历史有效信号数 `>= 2`

## 恢复版本

历史前端版本放在 `recovered_versions/`：

- `index_first_dark_20260508_before_110854.html`
- `index_first_light_20260508_110906.html`
- `index_first_light_completed_20260508_111750.html`

## 本地数据

`*.duckdb` 是本地缓存/数据库文件，不纳入 Git。
