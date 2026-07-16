# 步骤 2：检索、去重与失败记录

检索时间：2026-07-15（Asia/Shanghai）
复核时间：2026-07-16（Asia/Shanghai）

外部检索只使用公开、泛化后的主题词，没有发送未发表的 CCOD 完整措辞。三组实际查询为：

1. `Earth observation satellite scheduling learning action ranking solver schedules`（2000–2026）；
2. `budgeted counterfactual cost-to-go imitation learning large action space`（2000–2026）；
3. `conflict graph resource footprint diverse query selection combinatorial scheduling`（2000–2026）。

详细的逐查询高层结果、来源错误、候选合并和全文文件见 [`allinone.md`](./allinone.md)。

## 来源状态

| 来源 | 查询 1 | 查询 2 | 查询 3 |
|---|---|---|---|
| arXiv | 返回 10 条 | 返回 10 条，噪声较高 | 返回 10 条，几乎均为跨域噪声 |
| Crossref | 返回 10 条 | 返回 10 条，含一条反事实模仿学习近邻 | 返回 10 条，未发现 EOS footprint 直接先例 |
| OpenReview | HTTP 400，配置的用户名或密码无效 | HTTP 400，同上 | HTTP 400，同上 |
| OpenAlex | HTTP 401，未授权 | HTTP 401，未授权 | HTTP 401，未授权 |
| Semantic Scholar | HTTP 403，禁止访问 | HTTP 403，禁止访问 | HTTP 403，禁止访问 |
| DBLP | 返回 0 条 | HTTP 503 | HTTP 503 |

本机 `.venv` 首次调用还出现依赖不完整；随后使用 Homebrew 的 `/opt/homebrew/bin/python3` 重跑，才得到上述可复核输出。这是工具环境问题，不计入学术检索命中。

## 去重与补充策略

- arXiv 与 Crossref 的预印本/会议版本按题名、作者和 DOI 合并；Mercado-Martínez 2025 的 arXiv 与 ICMLCN 版本视为同一工作。
- API 排名噪声很大，不能用词面排名决定新颖性；最终高风险集合由机制轴匹配和全文证据决定。
- 为避免 API 漏召回，补入经典谱系 SeaRNN、LOLS、AggreVaTe，以及同域 Herrmann–Schaub、Jacquet；另加入 RILO 与 Mercado-Martínez 检验“异构动作空间”和“EOS 图动作评分”两条近邻。
- 七篇高风险候选均已取得全文文本并至少检查问题定义、方法和实验/局限三类证据。

## 初步去重结论

没有检索到一篇论文同时覆盖“EOS solution-induced traces + 约束一致 footprint + 固定反事实查询上限 + 容量平衡双尾覆盖 + train→dev 完整排程验证”。但通用的 rollout cost-vector、cost-to-go imitation、动作子采样和无搜索价值蒸馏均已存在，因此“没有完全相同论文”不等于 Level 1 新颖性。

补充定向检索还发现 Belaid 2026 的 *Optimizing Earth Observation Schedules under Unknown Operational Constraints: Active Constraint Acquisition*（arXiv:2604.13283）。它在 EO 调度中主动查询 feasibility oracle 以获取未知操作约束，是重要边界工作；但它查询的是“约束是否可行”，不是对已重建状态的备选动作做 $H$ 步 value query，也没有当前 CCOD 的固定 $B$ footprint allocation、balanced tail coverage 与等预算闭环收益链。本轮面向 budgeted counterfactual/action-value query selection 的检索未发现同时具备这四项的单篇工作；受接口失败和关键词召回限制，这只是当前证据集的保守结果，不是穷尽性证明。
