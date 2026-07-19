"""主力资金雷达。

候选池 = 细分行业龙头（stock_crawl_segment_leaders 选出、回写主库 sw3_member.is_leader）。
潜伏分目标：捕捉「左侧吸筹」——主力在低位悄悄建仓、但价格还没起飞的阶段。

  调研背景：「放量+创新高」经 verify 回测证明是右侧追高(截面 RankIC 显著为负)。吸筹的真正
  指纹是方向性的——参考 Wyckoff/VSA、A股筹码分布、OBV/ADL 三套体系，落地三个判别信号：

ambush 吸筹分（十三项 0~100 原始分直接加权）：
  筹码  低位筹码集中                                  权重 0.10
  位置  价格中低位                                    权重 0.10
  CMF   高买压反向有效分（最高50）                     权重 0.10
  P1    超额优先复合确认                               权重 0.10（双池10日有效，小盘更有效）
  P2    低位真实长下影温和缩量承接                     权重 0.05（大盘更有效）
  P3    缩量收盘大跌后首次完整收复                     权重 0.10
  P5    双底右腿确认                                  权重 0.05（临时核心有效，小盘更有效）
  P21   深破缩量收回                                  权重 0.10
  P23   低位压缩温和量能转强                           权重 0.05（核心有效）
  P24   OBV底背离连续确认后的首次成立                   权重 0.05
  P25   中低位缩量平台价格转强                         权重 0.05（双池10/20日正向，核心有效）
  户数  股东户数下降                                  权重 0.10
  回购  近90日公司回购                                权重 0.05
旧四技术因子加权分及其一字板/派发/P20折扣已删除；出货风险只在机会分中单独折扣。

另叠加「游资形态匹配」：把游资坐庄的「吸筹→试盘→
洗盘→突破→拉升→出货」六段套路编码成 match_patterns() 的布尔匹配器，给每只票打形态标签，再由
_pattern_phase() 汇总成一个主导阶段（详见下表 + 该函数 docstring）。

─────────────────────────────────────────────────────────────────────────────
游资形态总表（PATTERNS，6 类 / 26 个，编号 P1-P26）
  · 列：编号  名称  —— 命中条件（位置=收盘价近60日分位；量比=近5日均量/前20日均量；
        漂移=近20日涨跌幅；CMF=Chaikin资金流；筹码集中=主峰±7%价带内筹码占比）。
  · 信号方向 buy/hold/sell/observe；observe仅展示结构，不参与主导阶段。阶段配色按操作进程由早到晚渐变：
    吸筹🟢→试盘/洗盘🟡→突破/拉升🟠→出货🔴（观望⚪）。
  · 阶段优先级见 _pattern_phase：出货预警（P14/P15/P16/P17/P19/P20/P22/P26任一命中）> 突破 > 吸筹/洗盘 > 拉升 > 试盘。
─────────────────────────────────────────────────────────────────────────────
【吸筹 🟢buy】主力在低位悄悄建仓
  P1 超额优先复合确认  原低位缩量企稳连续2日首次确认，或低位首次突破前10日最高收盘
                       且通过510310五条件日终风险门；两腿任一命中即触发，小盘更有效
  P2 低位影线吸筹      位置<0.40 + 近10日真实长下影≥2根(下影≥2×实体且≥2%)
                       + 当日量能为此前20日均量的0.4~1.0倍，且当日自身再次出现长下影；大盘更有效
  P3 缩量打压首次收复  位置<0.32 + 近10日有"收盘跌≥4.2%且量<此前20日均量"，
                       期间收盘最多再破底6%，只在真正首次完整收复跌前收盘时确认
  P4 量增价稳观察      位置<0.60 + 量比1.2~2.5 + |漂移|<6% + CMF>0，首次进入时报告；
                       双池留出未验证为买点，不驱动绿色吸筹阶段
  P5 双底右腿确认      位置<0.45 + 两摆动低点(-2%~+10%) + 颈线反弹≥5% + 二底后8日内
                       收盘触及颈线，且由昨日未突破切换为今日突破近3日收盘高点：W底右腿事件；小盘更有效
  P23 低位压缩温和量能转强 位置<0.36 + 近20/60日振幅比<0.96 + 收盘≥1.005×MA20；
                       结构首次成立，且仅信号日量能/此前20日均量在0.4~1.2
  P24 OBV底背离        位置<0.50 + 30日价≤+3% + OBV净流入>0.10 + 背离>0.25，连续5日后首次确认
  P25 中低位缩量平台转强  120日中低位 + 60日横盘缩量 + 20日价格转正 + 临近20日平台
【试盘 🟡hold】拉升前试探上方抛压
  P6 试盘长上影        近8日有长上影(>3%且>2×实体)创20日新高后收盘缩回；保留形态观测，不作为有效因子
  P7 底部异动放量      位置<0.40 + 量比>1.5 + 近5日最大振幅>7%；保留形态观测，不作为有效因子
【洗盘 🟡buy】震仓甩浮筹、不破结构
  P8 缩量回踩洗盘      站上MA20 + 回踩近10日高点-2%~-15% + 跌日量<涨日量 + 筹码集中≥0.40：挖坑不破位
  P9 边拉边洗          多头排列(MA5>10>20) + 近8日涨跌符号切换≥4次 + 低点抬高：边拉边洗
  P10 高换手洗盘       量比>1.5 + 收盘>MA20 + 筹码集中≥0.45：高换手震仓但筹码峰不发散
  P21 深破缩量收回     近5日深破前40日箱底≥2.5%、收盘站回 + 当日量≤前20日均量0.75倍；双池10~40日正向
【突破 🟠hold（阶段标签，优先级仅次出货）】放量右进
  P11 放量突破启动     右进左出触发器命中：从低/中位放量(>1.3×)突破近20日收盘高点；全历史复测为负向风险
【拉升 🟠hold】超短动量，持有期拉长需防反转
  P12 连板拉升         连续涨停 streak≥2；复测 2 日动量有效，10~40 日转为反转风险
  P13 首板卡位         今日首板(近20日无涨停) + 换手10%~45%；复测 2 日动量有效
【出货 🔴sell】高位派发，当风控示警
  P14 高位放量滞涨     位置≥0.85 + 量比>1.5 + 近5日涨幅≤2% + 上影>2%；复测 40 日风控有效
  P15 新高量背离放量回撤 前15日出现旧P15新高量背离 + 距近10日盘中高点回撤≥4% + 单日量比>1.5
  P16 阴天量回撤确认   位置≥0.80 + 近3日出现40日最大量且收跌 + 距近10日最高价回撤≥3%；
                       全历史双池 2~40 日胜率与同池超额均低于旧版
  P17 新鲜倒V反转      位置≥0.80 + 冲高>16%后回撤≥8% + 峰值首次出现于最近5根 + 信号日不反弹；双池2~40日增强
  P18 顶部大阴包阳      位置≥0.80 + 昨阳今阴且今实体完全吞没昨实体：顶部看跌吞没
  P19 灌压巨量大阴      位置≥0.70 + 量比>1.8 + 实体跌>6%且收在当日价区下1/4；复测 5~40 日风控有效
  P20 均线放量破位      收盘跌破MA20(第5根bar仍在MA20上方) + 量比>1.3 + 当日跌≥1%；双池复测 2~40 日风控有效
  P22 放量假突破        今日盘中破前40日高但收盘没站上 + 当日放量>1.8×；全历史复测 2~40 日风控有效
  P26 获利盘冲高回撤风险 获利盘≥90%或5日前<40%且当前≥60% + 当日量≥前20日均量2倍
                       + 近5日涨≥5% + 收盘较近10日最高价回撤≥3%；双池覆盖与负向效果通过复测

当前跨池核心有效集（P15/P21于2026-07-15按双池复测加入，P5/P23/P25含产品口径决定）：
P1、P2、P3、P5、P11、P12、P13、P14、P15、P16、P17、P19、P20、P21、P22、P23、P24、P25、P26。
P6、P7 保留命中数据用于后续研究，但不作为有效因子或前端高亮。
P1/P3/P21各占吸筹分10%；P2/P5/P23/P24/P25各占5%。
─────────────────────────────────────────────────────────────────────────────

Modes:
  ambush   默认。给候选龙头算吸筹分 + 匹配游资形态，排名落盘。
  watch    实时交易监控入口（仍为骨架，后续接盘口/实时行情）。
  verify   吸筹分后验回测：PIT 重算分数 vs 未来前向收益(分位单调性/截面RankIC/多空价差/触发器超额)。
  patterns 形态预测力后验：每个形态命中后，未来 N 日相对同日全体的超额收益(剔大盘 beta)+t，
           验证哪些形态真有预测力(buy 看正超额、sell 看负超额)。

═══════════════════════════════════════════════════════════════════════════════
研究纪要（verify 回测沉淀，universe=细分龙头 / 区间 2023-2026 / 截面 Rank IC 为准）
═══════════════════════════════════════════════════════════════════════════════
口径说明：pooled 分位收益含大盘 beta（高分扎堆普涨日→假象）；**截面 Rank IC / 多空价差 /
触发器超额** 是每个 as-of 日内部排序再跨日平均，天然剔除 beta，才是真实选股力。
⚠️ 口径迭代：(1)-(15) 均为研究过程的历史记录；其中 (1)-(9) 还包含成交量单位混用等旧口径。
   当前 P1-P24 有效性统一以全历史逐日复测为准；统一版 P25 见 (22)。

迭代与结论：
  1) 放量+创新高（最初）  → 截面 IC 5/10/20日 -0.035/-0.039/-0.048 (t≈-3.6~-4.5)。
     = 右侧追高/见顶探测器；高分票未来 1-4 周反而走弱（短期反转）。和「天价见天量」自洽。
  2) 左侧吸筹 v1（量增价稳）→ 负 IC 砍半到 ~-0.02，仍未转正（量比本身就是反转特征）。
  3) 特征级诊断（10/60/120日）→ vol_ratio/振幅/动量/位置/换手 **全周期 IC 全负、6 个月不翻正**；
     龙头池被「短期反转 + 低波动异象」主导，没有任何「强势/放量」特征有正预测力。
  4) 三信号吸筹分 v2（位置+努力结果背离+CMF买压+低位筹码集中，本文件实现）→ 20/40/60日
     IC -0.001/-0.003/-0.017。**负 IC 被彻底中和到中性**（信号不再追高接盘），但没转正。
  6) cap-filtered(591只) + 右进左出突破触发器 →
       · 吸筹分截面 IC 翻微正：20/40/60日 +0.004/+0.006/+0.004（胜率达60%@40日），但 t<1 不显著；
       · **右进左出触发器失败**：触发样本绝对收益看着高(+1.31%>全体+0.85%)是 beta 假象，
         截面超额为负 -1.2%/-2.0%/-3.0% (t-1.3~-1.7)，胜率<50%。买突破=接盘，与(1)同源。
  7) v3 三角筹码分布 + 市值缺失用 stock_history 回补后，cap-filtered(582只) →
       · 吸筹分截面 IC 20/40/60日 +0.008/+0.005/-0.001，t<1，仍是微正/中性；（注：此为加出货惩罚前的基线，已被(9)改善）
       · 右进触发器继续负超额：40日 -2.0%(t-1.4)。结论仍是只看状态，不追突破。

核心结论：龙头池截面被**短期反转**主导——追强/突破/放量创高都亏，只有「安静的
吸筹状态」（低位+资金悄悄流入+不动）正向；叠加出货渐变惩罚后 20/40日截面 IC 显著(t≈2，见9)，
但多空价差仍≈0、Q5高分非最强，经济意义弱。量价 hot-money 命题在此池拿不到稳健
alpha。**根因疑为 universe 错配**：游资/主力打的是小盘低流通题材股，不是细分龙头；DB 目前只有
龙头 K 线（956 龙头+41 非龙头），无独立小盘池。下一步若要真正检验命题，需联网爬真·小盘/题材
股池在游资主场重跑 verify。当前代码=三信号吸筹分 + v3 三角筹码分布，作为研究基线沉淀。

  8) 【历史结果，已由(16)替代】游资形态层（P1-P20, match_patterns + patterns 模式）→ patterns 后验(582龙头/4.3万样本)：
     · 出货类经验证有效：P19 灌压巨量大阴 40日超额 -4.97%、P20 均线放量破位 -1.39%(t-2.16)。
       · P20 位置分桶：低位<0.6 命中930(占74%) 40日 -1.41%(t-2.03 显著)；高位≥0.6 仅321且不显著(t-0.85)。
         即「低位放量破位」=失败反弹/下跌中继才是 P20 主体信号，**不可按位置门控**(加门槛会扔掉74%有效信号)；
         故低位股命中 P20 仍是有效风险证据而非 bug——结构因子漏掉、形态层补上的真实风险；
         当前命中 P20 即直接升级为「出货预警」。
     · buy 类多为反转陷阱(P8 洗盘 -1.17%/t-3.0、P10 -1.61%/t-3.0)；唯 P3 缩量阴线打压吸筹
       +3.35%/60日+3.94%(t1.47 近显著)有正向苗头。拉升类 P11/P12 负超额，追入=接盘。
     · P5 底部形态构筑(双底/W底, 后补)→ 命中2080, 20/40/60日 -0.65%/-0.84%/-0.82%(t-1.46)，
       buy 形态却负超额=又一反转陷阱；同 universe 错配，识别本身没问题但龙头池无预测力。
     · P21 假跌破收回(Wyckoff spring, 沿用早期形态研究)→ 命中1624, 20/40/60日 +0.35/+1.23/+0.62%
       (t1.23)：buy 侧三档全正、t 比 P3 高且样本3×，是龙头池里统计最可靠的正向 buy 形态(虽未达 t1.5)。留用。
     · P22 放量假突破(高位拒绝, 同源)→ 命中517, 20/40/60日 -0.47/-1.11/-1.84%(t-0.83)：sell 看负、方向稳、
       胜率39%，未显著但作辅助风控标签留用。
     · 结论：出货预警层(P19/P20/P22)可当风控；buy 侧正向只有 P3/P21(含P5在内其余失效)——与全文一致。
     · 备注：形态只产出"阶段标签"、不进吸筹分加权(分数=4因子)，故失效形态无需"降权"，留作研究基线。
  9) 沿用早期形态研究的「出货分渐变惩罚」（吸筹分 ×(1−DIST_PENALTY_CAP×派发分/100)，
     派发分=高位×(基础+阶段涨幅+高位放量)）后重跑 verify（cap-filtered 582只）：
     · 吸筹分截面 IC 20/40/60日 +0.021/+0.017/+0.011，t 2.32/2.02/1.27——相比(7)的中性，
       20/40日 t 跳到 >2 显著正：把"高位假吸筹"剔出高分区、排序更干净（borrow 生效）。
     · 但多空价差仍≈0/负(+0.0%/-0.2%/-0.9%)、Q5高分非最强(40日 Q5+2.4% < Q4+3.5%)——RankIC
       转正靠中低分重排，不是"买最高分"能赚；定性为"无效→弱有效"，仍非可直接交易 alpha。
     · 触发器不受惩罚影响仍负超额(40日-2.0%/t-1.4)。
     · 试过(已弃)：给派发分加「高位放量长上影拒绝」维度→verify IC 0.021/0.017→0.0215/0.0179 几乎不动
       (只动高位股、而高位股吸筹分本就趋零)，增益≈0 故不保留。
 10) 【历史结果，已由(16)替代】成交量单位修复(全库统一手) + 前向 2/5/10/20/40日、逐周期判有效(剔大盘582只)：
     · verify 吸筹分截面 IC 2/5/10/20/40日 +0.012/+0.013/+0.019/+0.021/+0.017，t 1.28/1.46/2.04/2.33/1.92，
       胜率57/53/60/57/53%——IC 峰在 20日(t2.33)、10/20/40日均显著：**吸筹是 10-40日慢信号，2-5日弱**；多空价差≈0。
     · 触发器全周期负超额(再确认)：追突破任何持有期都亏，越长越亏(40日最负)。
     · patterns 逐形态×逐周期「有效@」(buy正/sell负/hold显著，均 |t|≥1.5)：
       - 风控 sell 多周期稳健：P22 放量假突破@5/10/40日、P20 均线破位@10/40日、P19 灌压大阴@5/40日(40日-4.97%)。
       - P11 放量突破(hold)@2/20/40日但**全为显著负**(追突破=亏，40日-3.42%)——量化"买突破=接盘"。
       - 动量：P12 连板拉升@2/5日显著正(+3.9/+3.5%)、20日反转-2.3%——连板短线续涨、3-4周见顶。
       - **buy/吸筹侧无一达标**(P3 40日+2.2%但t<1.5、P21 40日+1.15%弱、P5/P1/P8/P9/P10 多周期负)：龙头池买入侧仍无显著 alpha。
     · 结论：可用的只有**风控(P20/P22/P19/追突破P11)** + **超短动量(连板2-5日)**；吸筹买入侧不显著且偏慢(10-40日)，与全文一致。
     · 全市值(956只)个股 sell/风控形态更丰富：P11 追突破**全周期显著负**(40日-2.07%)、P14/P15/P16/P19/P20/P22 多周期✅、
       P3 缩量打压吸筹 40日 +4.21%*显著(全市值唯一达标 buy)、连板 P12 动量增至 2-10日。雷达默认含大盘全市值。
     · P23 箱体波动压缩(沿用早期形态研究的 compression, 后补)：命中众多但**各周期超额≈0、无edge**
       (剔大盘仅2日+0.18%*巨样本噪声)——又一买入侧失效形态，留作"酝酿"evidence标签/研究基线，不入有效集。
     · evidence 升级为结构化 {label,kind}(利好/中性/风险)，前端按 kind 上色(替原关键字正则)。
 11) 因子拆解 + 背离/CMF 反向计入（剔大盘，子分截面 IC）：
     · 单因子 RankIC(t)：**位置 +0.05/t3~4.5(唯一强正)**；背离 -0.04/t-3.6、CMF -0.02/t-1.8~-2.1(均显著负，
       "放量价稳/收盘买压"在龙头池是反转特征)；筹码 ≈0/噪声。合成分原本靠位置一个扛、被背离+CMF 抵消。
     · 改动：**背离、CMF 改"反向计入·只罚高不奖缺失"**：贡献=min(50,100−原始分)、权重不变；筹码/位置不动。
       (纯 100−x 会把"无放量/无买压"翻成满分→死水股被抬，故封顶 50 中性；两者 IC 等效。)原始子分仍存 sub_scores。
     · verify(剔大盘,真ambush含惩罚) 2/5/10/20/40日 IC ~+0.04(t≈3)、胜率60~68%、**多空价差转正**——相比翻转前
       (0.012~0.021/t≈2、多空负)，IC 翻2~3倍、多空首次转正。
     · 保留：本质仍是「位置=低位反转」主导、背离/CMF 反算只是去拖累；**样本内、未做 OOS**，是反转 alpha 非真吸筹。
 12) P3/P20 消融(吸筹分加 P3 加分 / P20 破位惩罚, ±15分, 前后半 OOS)：
     · **P20 破位惩罚有增量**：全周期 +0.004~0.005 IC、前后半都成立(纠错型——把"低位破位但分仍偏高 0.60 分位"的票挪下，
       补 dist 高位门控的盲区)；60日外失效(破位是近中期信号)。→ 已加 `MA_BREAKDOWN_PENALTY=0.20`(命中打掉20%)。
       实测加后 IC 2/5/10/20/40 ≈ 0.049/0.036/0.039/0.041/0.047、t2.7~3.7、多空价差全正。
     · **P3 加分无增量、弃**：全周期≈0、60日转负。原因(方法论)：① P3 稀疏(1.5%)，加分只动 1.5% 名次→截面 IC 不变；
       ② **冗余**——P3 要 pos<0.45，命中票本就被位置/筹码因子排到高位(分位均值0.65)，再加分不改排序；③ 均值 vs 排序(超额是均值、可能偏度驱动)。
       **教训：事件研究(patterns)超额高 ≠ 进合成模型能加分；判该不该进模型看「边际 RankIC 贡献(消融)」，而非「条件超额」。**
     · ⚠️ **edge 时间不稳**：吸筹分 IC 几乎全在后半——前半 5/10/20/40日 t 仅 0.1~0.3(近零)、后半 t 3~5。
       所有"IC≈0.04/t3"由近段单边撑起，可能 regime 依赖，**勿过度外推**。
 13) 数据层扩张（价量代理不了的"谁在吸筹"——独立爬虫入库 stock_data.sqlite3，剔大盘验证）：
     先验：RS/OBV 等价量 buy 因子全败(重新发现反转，见纪要附测)；遂补基本面/资金/内部人数据层。
     · ✅✅ **股东户数下降**(shareholder_count, stock_crawl_holders.py)：因子=−户数增减比例。
       截面 IC 2/5/10/20/40日 +0.022/+0.018/+0.032/+0.037/+0.047，t 3.0/2.2/3.8/4.6/5.9，全周期显著、随期递增。
       与吸筹分相关 ρ≈-0.06(近正交)、与位置 ρ≈-0.06 → 真新 alpha 非反转代理。等权合成"吸筹分⊕户数"
       IC 提到 0.05~0.067、t 5~7.2(1+1>2，因不相关降噪)。时间剖面与吸筹分**互补**(户数前半强/吸筹后半强)。
       **目前最佳买入侧信号**(慢/季度/结构)。
     · 🟡 **回购**(repurchase)：事件研究 20/40日 +1.38%/+1.96%(t1.8/1.7)近显著正——干净的公司行为、无追高污染，弱买点。
     · ❌ **增持**(holder_increase)：5/10/20日 -0.4~-0.7%(t≈-1.5~-1.8)弱负——多为下跌护盘，弃。
     · 🔴→风控 **龙虎榜**(lhb_all 全榜, stock_crawl_capital.py)：命中后全周期显著负(40日-1.55%~-2%/t-2.9~-3)。
       全榜>0/任意上榜/机构净买 三口径**方向一致全负**，全榜样本大更显著→「上榜=异动追高」系统性反转。
       **是反向/避雷信号，非买点**(与 P11 追突破、机构净买同源)。已弃机构专表 lhb_inst，统一用全榜 stock_lhb_detail_em。
     · 大教训：「资金/内部人」信号也分两类——**绑在异动/公告事件上的(龙虎榜/增持)被反转污染甚至反向**；
       **慢结构(股东户数)或纯公司行为(回购)才干净有效**。"机构/大单在买"≠"该买"，因为它们也在追高。
       全文主线再确认：瓶颈是**缺数据层**(户数一上来就 t5.9)，不是缺聪明价量因子。
 14) universe 扩张（游资小盘池 is_hot_money）+ 超短反转分（hotmoney 池打分，_apply_reversal_model）：
     验证"龙头池错配、游资主场是小盘题材股"猜想→建池：近1年龙虎榜≥5次 ∩ 流通市值≤100亿 ∩ 非ST非次新
     = 541只(stock_crawl_hot_money_universe.py 建、标 is_hot_money)，其上重跑超短(前向1/2/3/5日)：
     · ❌ 动量证伪：游资池反转比龙头池更猛——换手分位 IC3d -0.10(t-7.5)、当日振幅 -0.105(t-6.4)、
       近5日涨停 -0.066(t-6.8)、量比/动量全显著负(t-4~-10)；连板 streak T+1 +0.05(t6.5)但 T+3 转负。
       追涨/放量/高换手在游资小盘是更致命的短期反转陷阱(强度 2-3× 龙头池)。
     · ✅ 反转可交易：反转分(过热因子反向加权 换手.30/振幅.20/涨停.20/量比.15/动量.15)
       verify 2/3/5日 IC +0.107/+0.101/+0.125(t6.6~9.6)、多空 +0.51~+0.68%/3日——**全研究首个多空转正**，
       强度远超龙头池吸筹分(0.05)/动量(0.035 多空≈0)；与吸筹分 ρ0.40 半独立。
     · ✅ 市场状态择时(沪深300, _market_regime)：反转分 top20% 仅在大盘>MA20 时做多→3日 +0.83%(基线+0.47%翻倍)、
       胜率53%；<MA20 仅 +0.13%(接刀)。择时主要提收益非胜率(固化版5因子含超跌 mom_5d，基线胜率已 52%>50%)。
       规则=反转分选股 + 大盘站上MA20 才做多；信号日大盘跌(ret1<0)反更好(+0.62%/54%，恐慌杀跌后反弹)。
     · 落地：radar --pool hotmoney 主排序=reversal_score(过热因子反向)、payload 附 market_regime(读 index_nav 510310
       判大盘>MA20=是否适合做多)；leader 池仍用机会分。反转分=排序/多空信号，纯做多须配 regime 择时。
 15) 反转分时间平滑（2026-06 重建 hotmoney 池 543 只重测；REVERSAL_SMOOTH_DAYS=3=ema3）：
     问题=单日快照 vs 过去 N 日各算一次截面反向 rank 子分再 EMA(今日权重最高)。消融(驱动生产
     _apply_reversal_model，切 1↔3；140 截面/2023-07~2026-06)：
     · 全样本：单日 ≈ ema3(IC 打平,3日~0.11/t11)，平滑仅把多空价差抬一丢丢(3日+0.61%→+0.68%/10日+1.38%→+1.48%)。
     · **favorable(大盘>MA20,真做多场)平滑明显占优**：3日 IC 0.107→0.115(t7.1→7.7)、多空 +0.44%→+0.53%；
       5日 0.095→0.104、多空 +0.41%→+0.61%；10日多空 +0.96%→+1.15%。
     · OOS 时间分半会**变号**：前半(强 regime)单日略优(平滑稀释最新极值,3日 0.131 vs 0.130 ~平)；
       后半(信号衰减/更近未来)平滑全周期反超(3日 0.088→0.094、10日 0.099→0.107、多空抬升)。
     · 结论：平滑=**稳健补丁(二阶,多空 ~+0.1~0.2%/截面)**，单日信号越噪越值钱；ema3 前半几乎不掉、
       近段/favorable 吃增益，故选 ema3 默认(ma5 增益更大但前半短周期代价也大)。坐实"反转溢价归属
       *持续过热*的票、窗口比单日排得更干净"。REVERSAL_SMOOTH_DAYS=1 即关闭(退回单日，与旧版逐分等价)。
 16) 【历史稀疏抽样结果，已由(18)替代】leader 977 只 + hotmoney 545 只，75 个历史截面，
     区间 2023-04-04~2026-04-27，90 根 PIT 回看，前向 2/5/10/20/40 日，相对同日池均值；
     显著性采用 Newey-West HAC，并在每个池的 24×5 次检验上做 BH FDR 10%，另按时间三等分检查稳定性。
     · 当前有效形态仅为：P12、P13（超短动量）与 P14、P17、P19（出货风控）。
     · P12：双池 2 日超额 +4.86%/+3.86%；hotmoney 10/20/40 日转负 -3.90%/-6.32%/-9.98%。
     · P13：hotmoney 2 日超额 +4.69%，属于首板超短动量，持有期拉长后优势衰减。
     · P14：hotmoney 40 日超额 -4.24%；P17：hotmoney 2/5/10/20/40 日均为负；
       P19：hotmoney 5/10/20/40 日超额 -3.31%/-6.20%/-5.88%/-7.28%。
     · P3/P11/P20/P22 等旧有效结论未通过本轮统一复测，不再进入 PATTERN_EFFECTIVE；这不等同于删除
       P20 在复合吸筹分中的消融惩罚或 P14/P16/P17/P19/P20/P22 在出货分模型中的权重，两者检验问题不同。
     · 局限：当前成分股口径存在幸存者偏差；收益为收盘到收盘，未计交易成本、涨跌停与可买性。
 17) 【2026-07-11 P25 买入点研究 v3；hotmoney 池实验接入】双池并集 1457 只、198.1 万个可研究锚点；目标为信号日后
     5日累计涨幅>15%。形态强制为此前60日长期低位横盘、成交量水平收缩且趋势下降；另审计当日量价启动。
     训练≤2023、2024验证、2025二次验证、2026最终留出，全部只用当时可见数据。
     · 纯60日底盘：13,698个信号，5日上涨胜率49.79%，平均超额-0.25%；涨超15%命中率0.29%，仅为
       同期基准0.13倍，说明“低位横盘缩量”本身是蓄势状态而不是即时买点。
     · 加“当日量≥此前20日均量1.2倍、当日涨4%~6%”后：总体271个信号，胜率55.72%、平均超额+1.25%，
       涨超15%命中率4.43%（1.99倍基准）；但训练胜率仅52.82%，2025/2026均未命中涨超15%事件，
       2026的100%胜率只有10个信号，样本过小；leader池胜率54.35%也未达55%。
     · 连续模型总体胜率51.55%，2026为46.65%，不适合直接上线。进一步将选参限制在 hotmoney 池，加入
       “180日低位、近5日CMF≥0.20、当日量能≥前20日均量1.5倍、距20日平台不超过2%”确认。
     · hotmoney 分段胜率：训练55.42%、2024验证82.46%、2025二次验证83.33%、2026留出70.00%；池内合计
       63.67%，平均5日超额+2.35%，涨超15%的 lift=2.63。但 2025/2026 仅 12/10 个信号，且无>15%命中。
     · 结论：P25 在所有股票池统一计算，但只有 hotmoney 池有较好的分段证据，且不纳入 PATTERN_EFFECTIVE。
       继续用未来新数据滚动监控，样本外衰减时直接下线。
 18) 【2026-07-11 P1-P24 全历史逐日滚动复测；P23/P24优化前口径】事件研究不训练模型，故不划分训练/验证区间，
     每只股票在满足 180 根回看与 40 日前向收益后逐交易日产生样本。数据库可用区间 2017-03-16~2026-05-14；
     leader 602 只、2223 个交易日截面、1,146,047 个股票-日期样本；hotmoney 549 只、481,791 个样本。
     收益仍相对同日池均值；Newey-West HAC 按重叠周期使用 1/4/9/19/39 阶滞后，并在每池 24×5 次检验上做 BH FDR 10%。
     · 当时核心有效集：P11/P12/P13/P14/P16/P17/P19/P20/P22；P24优化后结论见(20)。
       P6/P7 虽保留历史命中与计算链路，但不再作为有效因子或前端高亮。
     · P12/P13 是方向随周期变化的动量信号：P12 双池 2 日 +3.41%/+2.84%，P13 +1.13%/+2.14%；
       但 hotmoney 池 P12 10/20/40 日 -3.22%/-5.53%/-8.33%，P13 20/40 日 -2.33%/-4.06%。
     · 最强风控为 P19（hotmoney 2~40 日 -2.31%~-5.90%），其次 P17、P16、P22、P14；
       P11/P20 也是双池负向风险。P6/P7 的信号较弱且稳定性不足，降级为观察形态。
     · 单池次级信号：旧P2/P24仅hotmoney中长期弱正；旧P15/P18仅hotmoney 20日负向。
       它们统计显著但跨池证据不足，不进入 PATTERN_EFFECTIVE。
     · 局限：当前成分股口径存在幸存者偏差；收盘到收盘收益未计交易成本、涨跌停与可买性。
 19) 【2026-07-11 P2阈值专项】原“15日≥3次”命中 leader/hotmoney 39.70万/14.63万，命中率34.64%/30.37%。
     网格覆盖 X=5/8/10/12/15/20/25/30、Y=2..X，并比较状态命中/当日再次确认/当日严格长下影三种模式；
     本研究不训练模型，只按2017-2021早期、2022-2023中期、2024+近期机械分段检查稳定性。
     任何X/Y都不能令P2在leader池成为稳定买点，但因子仍在所有池统一计算。
     hotmoney“近10日≥5次”状态规则命中18,639、命中率3.87%；后续按使用要求增加“当日也必须命中”，变为10,135次、
     命中率2.10%，全样本10/20/40日超额+0.13%/+0.49%/+0.82%。该触发式规则跨时段证据较弱，继续作为实验标签。
 20) 【2026-07-11 P23/P24低频确认优化】双池全历史网格比较严格阈值、连续3/5日确认与一次性触发；所有池参数相同。
     · P23 严格候选曾加入振幅比<0.65、20日箱体<15%、20/60日量能比<0.90及连续5日首次确认，命中降至
       1,178/577，但HAC/FDR仍未通过。按当时使用要求，生产P23曾恢复原“位置<0.60 + 振幅比<0.80”
       状态规则；当前事件化版本见(29)。
     · P24 改为位置<0.50、30日涨幅≤3%、量纲化OBV>0.10、OBV与价格背离>0.25，并连续5日后首次确认。
       命中由138,093/65,937降至1,144/677；10日超额+0.67%/+0.82%，绝对胜率56.5%/64.5%，双池10日
       均通过HAC与BH-FDR 10%，因此加入 PATTERN_EFFECTIVE。
  21) 【2026-07-11 P3首次收复优化】旧P3把信号日附近30日均量用于更早的下跌日，且同一事件在收复后跌回、
     再次收复时仍可能重复命中；
     改为下跌日只和此前20日均量比较，并要求8日内首次完整收复跌前收盘、期间不深破底、确认日不过度放量。
     参数只用2017-2023开发段排序，2024+留出不参与选参；leader/hotmoney 两池始终使用同一规则。
     · 生产参数：位置<0.35、跌幅≥4.5%、下跌量比<0.85、8日内完整收复、最多再破底3%、确认日量比≤1.30。
     · 全历史命中843/605；2/5/10/20/40日超额 leader +0.44/+0.44/+0.64/+0.65/+0.37%，
       hotmoney +0.46/+1.04/+0.70/+0.92/+2.36%，十个池×周期全部为正。
     · leader 2日与 hotmoney 5日通过各池24×5次检验的BH-FDR 10%，因此P3重新加入 PATTERN_EFFECTIVE；
       仍存在当前成分股幸存者偏差，且未计交易成本与涨跌停可买性。
 22) 【历史结果，已由(33)替代；2026-07-12 P25统一升级】按使用决定，将严格缩量底部方案作为全池统一 P25，不再区分
     leader/hotmoney 公式，并删除沪深300站上MA20门槛。判据为120日位置≤45%、60日振幅≤25%、60日绝对涨跌≤8%、近20日/前40日均量≤70%、
     60日量能斜率≤-1%、当日量≥前20日均量1.5倍、距20日平台≥-2%；保留防追高门槛。
     · leader 1,296次：2/5/10/20/40日同日池超额 -0.11%/-0.10%/-0.34%/-0.58%/-1.77%。
     · hotmoney 282次：2日约0%，5/10/20/40日 +0.39%/+0.63%/+1.51%/+1.46%，但FDR均未通过。
     · 结论：P25在所有池统一计算，继续作为实验标签，不加入 PATTERN_EFFECTIVE。
 23) 【历史P25结果，已由(33)替代；2026-07-13 P1/P25双池计分】统一180日回看复测显示，两者在leader池各周期同池超额均为负；
     hotmoney池P1 5/10/20日超额+1.20%/+1.08%/+1.54%，P25 10/20/40日+0.57%/+1.36%/+1.33%。
     当时按产品决定仍让P1/P25在leader/hotmoney双池触发并各占吸筹分5%，前端和目录曾标注“仅小盘有效”；
     筹码集中权重由20%降至10%，其余七项权重不变。P1/P25不加入跨池PATTERN_EFFECTIVE。
 24) 【2026-07-13 P1简化扩容】当前hotmoney成分全历史复测：删除无增量的20日收盘区间门、现价高于
     筹码峰上限和获利盘上限，并把60日最高/最低振幅由≤25%放宽至≤30%；三日确认、市场门和其余保护门不变。
     · 10日事件117→147（+25.6%），信号日99→123（+24.2%），胜率59.0%→63.3%，同池超额
       +1.13%→+1.44%，HAC t 2.12→2.67；2017–2022、2023–2024、2025+三段均为正超额。
     · 获利盘下界继续保持20%；降到15%虽可把事件扩到170，但早期段超额由+0.46%降至+0.18%，不取。
     · leader池不满足扩容守门：range放宽后10日胜率下降1.62pct、40日负超额加深，故P1仍只标“小盘有效”。
 25) 【2026-07-13 P2真实长下影+量能优化】删除“普通十字星也算影线”的旧口径，统一改为位置<0.40、
     近10日真实长下影≥2次（下影≥2×实体且≥昨收2%，当日必须命中），只叠加当日/此前20日均量0.4~1.0。
     · leader/hotmoney命中7,118/2,604，命中率0.386%/0.532%；10日超额+0.459%/+0.457%，
       绝对胜率54.58%/60.60%；20日超额+0.827%/+0.915%，绝对胜率54.73%/61.94%。
     · 同一2%影线阈值下，测试的12组相邻量比区间全部保持双池10/20日正超额；1.8%/2.0%/2.2%
       三档影线阈值在量比0.4~1.0时也全部为正。该方案仍是全样本探索结果，不宣称严格样本外。
     · 2日leader仍为负超额，P2定位为5~20日承接观察信号；本轮研究完成时暂未加入有效集或吸筹总分。
 26) 【2026-07-13 P2生产计分】按产品决定将P2加入PATTERN_EFFECTIVE并在前端使用绿色bullish样式；
     吸筹模型的价格位置权重由20%降至10%，腾出的10%全部分配给P2。其余八项权重不变，总权重仍为100%。
  27) 【2026-07-13 P3严格首次+降耗优化】修复旧规则在“收复→跌回→再收复”时对同一打压事件重复
     触发的问题；当前规则为位置<0.32、近10日收盘跌≥4.2%、事件量比<1.00、最多再破底6%、
     确认量比≤1.35，并要求事件后到昨日从未收复，今日才真正首次完整站回跌前收盘。
     · leader命中1,854/1,845,270（0.1005%），2/5/10/20/40日同池超额
       +0.41%/+0.74%/+1.30%/+1.29%/+1.64%，胜率53.5%/53.1%/59.4%/58.7%/54.3%。
     · hotmoney命中980/489,420（0.2002%），同池超额+0.76%/+1.53%/+2.23%/+2.47%/+4.02%，
       胜率55.8%/54.6%/59.6%/63.1%/52.4%；2024+收益在参数冻结后揭盲，双池10日超额仍为正。
     · 复用上下文确认量比、用收盘直接推导跌幅，并先做O(1)价格短路、只对候选跌日计算量均；
       虽把事件窗从8日放宽到10日，隔离matcher微基准仍明显快于旧生产实现。
 28) 【2026-07-13 P1双池替换】旧“长期底盘+筹码+510310市场门”覆盖不足且leader超额为负，生产改为
     pos60<40%、|ret20|≤4%、收盘/MA20≥0.995、收阳、当日量能/此前20日均量0.4~1.0，连续2日后首次确认。
     删除120日分位、60日振幅/收益、波动/低点路径、筹码和市场门；leader/hotmoney始终使用同一规则。
     · 全历史命中4,185/1,007，覆盖率0.227%/0.206%；10日胜率55.03%/55.51%，同池超额
       +0.46%/+1.19%，HAC t=2.79/2.38。
     · ≤2024开发段双池胜率53.41%/52.66%、超额+0.48%/+1.49%；2025+时间留出胜率
       63.61%/63.92%、超额+0.37%/+0.23%。留出HAC尚未显著，继续前瞻观察。
 29) 【2026-07-13 P23低位压缩事件化】原状态型P23覆盖leader/hotmoney 11.34%/13.62%，leader各周期
     超额为负。约1,300个不超过两个附加门的候选统一比较后，保留“位置<0.40、振幅比<0.80，
     核心状态由昨日不成立转为今日首次成立，且收盘≥MA20”；两个池使用同一规则，主持有期10日。
     · leader命中5,160/1,845,270（0.280%），10日胜率50.2%、同池超额+0.17%、HAC t=1.05；
       hotmoney命中1,623/489,420（0.332%），10日胜率51.8%、同池超额+0.22%、HAC t=0.83。
     · ≤2023/2024+的10日超额leader为+0.10%/+0.37%，hotmoney为+0.24%/+0.18%，四段胜率均>50%；
       但双池HAC均未显著，且leader 20/40日超额为负，故P23仅作10日实验信号，不加入PATTERN_EFFECTIVE。
     · 相比原规则只收紧既有位置阈值并新增“首次进入、站上MA20”两项；仍为固定窗口O(LOOKBACK)，
       不增加数据源、筹码重建或数据库查询。研究结论存在当前成分股幸存者偏差，且未计交易成本与可买性。
 30) 【2026-07-13 P4双池否定与中性化】原P4覆盖leader/hotmoney 2.480%/2.236%，2/5/10/20/40日
     同池超额全部为负；10日胜率49.0%/49.6%、超额-0.38%/-0.16%。按≤2023开发、2024验证、
     2025+最终留出冻结参数，在原四变量与最多两个低成本门内，没有方案同时满足三段双池胜率>50%、
     超额>0和0.1%~2%覆盖率，故不强选买入版本。
     · 生产仅修复规格漂移：量比恢复为(1.2,2.5]，核心状态只在False→True首次进入日报告；命中率
       1.174%/1.057%，但10日超额仍为-0.40%/-0.16%，因此signal改为observe，不进入有效集或吸筹分，
       也不再单独驱动绿色吸筹阶段。完整研究见meta_data_backup/p4_factor_optimization.md。
     · 昨日P4只重算固定180日价量上下文，不建筹码、不查新数据，复杂度仍为O(LOOKBACK)。
 31) 【2026-07-13 P5双底右腿确认】旧P5是二底后可连续多日成立的宽松回升状态，
     leader/hotmoney命中率5.163%/5.550%，10日胜率46.4%/47.6%、同池超额-0.35%/-0.10%。
     生产保留W底主体，把二底区间调为-2%~+10%、颈线反弹调为≥5%，收盘收紧到颈线的100%~103%；
     只额外增加“二底在近8日”和“昨日未突破→今日突破前3日收盘高点”两项，并删除筹码门。
     · leader命中2,142/1,845,270（0.1161%），10日胜率55.2%、同池超额+0.35%、HAC t=1.34；
       hotmoney命中641/489,420（0.1310%），10日胜率63.8%、超额+0.84%、HAC t=1.58。
     · 两池早晚分段的10日胜率与超额均为正，但双池HAC/BH-FDR未过门槛，故只作10日观察信号，
       不加入PATTERN_EFFECTIVE或吸筹分。固定60日收盘扫描不增加复杂度量级，详见p5_right_leg_research.md。
  32) 【2026-07-13 P5生产计分】按产品决定将P5作为5%原始特征加入吸筹分，同时把P3由20%降至15%；
     其余权重不变，总权重仍为100%。P5继续保留观察信号与非PATTERN_EFFECTIVE状态，计分不改变统计验证标签。
 33) 【2026-07-13 P25双池低频扩容】旧P25用“当日量≥前20日均量1.5倍”确认启动，双池命中率仅
     0.071%/0.059%，且leader 10/20日同池超额-0.31%/-0.52%。受限研究只替换两项确认：删除爆量门，
     改为20日价格涨幅≥0.5%；距前20日最高收盘由-2%放宽为-4%。底盘既有阈值同步放宽为120日分位≤65%、
     60日振幅≤35%、|60日涨跌|≤16%、量能斜率≤-0.5%，近20/前40均量≤70%保持不变。
     · leader命中24,466/1,845,270（1.326%），10/20日胜率52.4%/53.1%，超额+0.24%/+0.46%；
       hotmoney命中6,136/489,420（1.254%），胜率53.7%/53.9%，超额+0.55%/+0.86%。
     · ≤2023与2024+四个池段的10/20日胜率和超额均为正；ret20取0%~0.5%、平台距离取-4%~-3%的
       六个邻域也保持方向不变。hotmoney 10/20日HAC t=2.08/1.99，leader仅1.35/1.36，且逐年仍有波动，
       因此P25继续作为双池实验形态、主持有期10~20日，不加入PATTERN_EFFECTIVE。
 33) 【2026-07-13 P5临时有效】按产品决定暂时将P5加入PATTERN_EFFECTIVE并使用bullish样式；
     5%吸筹权重和10日主持有期不变，历史显著性限制仍保留在研究记录中。
 34) 【2026-07-14 P23覆盖与质量二次优化】以当时生产P23（位置<0.40、振幅比<0.80、核心状态首次成立且
     收盘≥MA20）为严格基线；新规则改为“位置<0.36、振幅比<0.96、收盘≥1.005×MA20”结构状态
     False→True，再只对信号日加量能/此前20日均量0.4~1.2；量能不参与状态去重。
     · 冻结快照生产等价复核：leader 12,461/1,837,526（0.6781%），10日胜率51.83%、同池超额
       +0.3315%、HAC t=2.62；基线为0.2796%/50.16%/+0.1541%/0.97。
     · hotmoney 3,715/496,233（0.7486%），胜率52.19%、超额+0.5290%、HAC t=2.66；基线为
       0.3319%/51.55%/+0.1825%/0.70。双池覆盖均进入0.5%~2%，全历史胜率和超额均严格提高。
     · hotmoney 2024+分段胜率52.26%→51.61%，虽超额+0.13%→+0.69%，仍须前瞻观察；未做完整FDR重验，
       P23继续作10日实验信号。仅新增一个低成本当日量能门，复用已有上下文，仍为O(LOOKBACK)且无新查询。
 35) 【2026-07-15 P21深破缩量收回】旧规则只要求近5日跌穿前40日箱底1.5%并站回，是可连续命中的宽状态；
     leader/hotmoney覆盖3.907%/4.172%，10日胜率53.24%/53.88%，但同池超额-0.102%/-0.210%。
     新规则把最低跌破收紧到2.5%，只叠加“信号日量能/此前20日均量≤0.75”一个既有指标；量能优先换手率，
     近180日换手覆盖不足70%时退回成交量。它识别深破后的缩量收回，避免把普通反弹与放量真破位混入。
     · leader命中11,402/1,828,471（0.624%），10/20/40日胜率54.53%/56.44%/54.45%，同池超额
       +0.332%/+0.510%/+1.346%；hotmoney命中4,566/500,359（0.913%），胜率56.44%/63.78%/60.91%，
       超额+0.421%/+1.327%/+1.949%。开发段与2024+的双池10/20日胜率、超额均为正。
     · False→True事件化会把覆盖压到0.259%/0.405%，低于0.5%业务下限且近期小盘超额转负；限制最大跌破5%
       也会令leader覆盖降到0.486%。故保留最多5日的状态口径且不设最大深度。P21加入核心有效集，继续只作
       洗盘阶段标签/买点，不改变吸筹总分权重；固定窗口与既有量能上下文使复杂度仍为O(LOOKBACK)。
 36) 【2026-07-17生产展示与权重调整】按产品决定，因子描述与命中形态解释明确标注P1/P5小盘更有效、
     P2大盘更有效；吸筹分中P5由10%降至5%，P21由5%提高到10%，其余权重不变，总权重仍为100%。
 37) 【2026-07-19 P23/P25生产有效性口径】按产品决定，将P23/P25加入PATTERN_EFFECTIVE并使用bullish样式；
     两者各占吸筹分5%、既有判据与历史回测数据不变，研究显著性限制继续保留在上述历史记录中。
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ProcessPoolExecutor
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

import stock_storage
from stock_etf_pool import load_etf_pool


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CAPITAL_DIR = DATA_DIR / "capital"
DB_FILE = stock_storage.DEFAULT_DB_FILE
ETF_REFRESH_REPORT_FILE = DATA_DIR / "etf_pool_refresh_report.json"
AMBUSH_RESULT_FILE = CAPITAL_DIR / "hot_money_ambush.json"
THEME_CANDIDATES_FILE = CAPITAL_DIR / "theme_candidates.json"   # stock_theme_candidates.py 落盘
WATCH_STATE_FILE = CAPITAL_DIR / "hot_money_watch.json"
VERIFY_RESULT_FILE = CAPITAL_DIR / "hot_money_verify.json"
PATTERNS_RESULT_FILE = CAPITAL_DIR / "hot_money_patterns.json"
DIST_EXPERIMENT_FILE = CAPITAL_DIR / "hot_money_distribution_experiment.json"
ACCUM_EXPERIMENT_FILE = CAPITAL_DIR / "hot_money_accumulation_experiment.json"
LATENT_RESULT_FILE = CAPITAL_DIR / "hot_money_latent.json"   # latent 模式：潜伏妖股观察名单
SCHEMA = "hot_money_radar.v3"
MODES = ("ambush", "watch", "verify", "patterns", "distribution", "accumulation", "latent")
DEFAULT_MODE = "ambush"

# ── 打分参数（集中放置，便于后续调参/优化器接管）──────────────
LOOKBACK = 180           # 统一历史窗口；P25判据使用其中近120/60日，P26另取5日前筹码
MIN_BARS = 40            # 少于这么多有效 bar 视为数据不足，不打分
SHORT_WIN = 5            # 近端放量窗口
BASE_WIN = 20            # 量比基线窗口（紧邻近端窗口之前）
HIGH_WIN = 60            # 价格分位窗口
DRIFT_WIN = 20           # 努力与结果背离：看近 20 日累计涨跌幅是否横住
VOL_RATIO_FULL = 2.5     # 量比达到该值给满分（1.0→0 分，线性）
POS_LOW = 0.25           # 收盘价分位 ≤ 此值 → 中低位满分（留足上行空间）
POS_HIGH = 0.70          # 收盘价分位 ≥ 此值 → 已拉升，位置分 0
CMF_WIN = 20             # Chaikin Money Flow 窗口
CMF_FULL = 0.20          # CMF 达到该值给满分（≤0 → 0 分）
CHIP_BUCKETS = 80        # 成本分布价格网格数
CHIP_DECAY = 1.0         # 历史换手衰减系数（n=1：今日换手多少就搬移多少昨日筹码）
CHIP_BAND = 0.07         # 主峰 ±7% 价带内筹码占比 = 单峰密集集中度
CHIP_CONC_LO = 0.25      # 集中度 ≤ 此值 → 0 分
CHIP_CONC_HI = 0.60      # 集中度 ≥ 此值 → 满分
CHIP_PEAK_FULL = 0.35    # 主峰处在近端价格区间低 35% 内 → 低位满分
CHIP_PEAK_ZERO = 0.70    # 主峰处在近端价格区间高 70% 以上 → 低位 0 分
CHIP_PRICE_BELOW_ZERO = -0.08  # 当前价低于主峰太多，说明尚未站回成本区
CHIP_PRICE_ABOVE_FULL = 0.08   # 当前价略高于主峰内仍视为吸筹未充分拉升
CHIP_PRICE_ABOVE_ZERO = 0.22   # 当前价高于主峰过多，视为已启动
CHIP_WINNER_MIN = 0.15         # 获利盘过低，上方套牢压力重
CHIP_WINNER_FULL_LO = 0.25
CHIP_WINNER_FULL_HI = 0.60     # 获利盘中低位最佳：不是全套牢，也不是满获利
CHIP_WINNER_MAX = 0.82
CHIP_WINNER_RISK_HIGH = 0.90       # P26：绝大多数筹码已获利，兑现/派发风险
CHIP_WINNER_RISK_DAYS = 5          # P26：与 N 个交易日前比较
CHIP_WINNER_RISK_PRIOR_MAX = 0.40  # P26：N 日前获利盘仍低于 40%
CHIP_WINNER_RISK_CURRENT_MIN = 0.60  # P26：当前快速跃升到 60% 以上
CHIP_WINNER_RISK_VOLUME_BASE_WIN = 20  # P26：当日量能对比此前 N 日均量
CHIP_WINNER_RISK_VOLUME_MIN_OBS = 15   # P26：此前窗口至少 15 个有效量能点
CHIP_WINNER_RISK_VOLUME_RATIO_MIN = 2.00  # P26：获利盘风险必须有极端放量确认
P26_RET_5_MIN = 0.05              # P26：先有近5日快速拉升，才把获利盘拥挤解释为冲高风险
P26_PEAK_HIGH_DAYS = 10           # P26：价格走弱确认所用近端盘中高点窗口（含信号日）
P26_PEAK_DRAWDOWN_MIN = 0.03      # P26：当前收盘至少较近10日最高价回撤3%
SEALED_AMP = 0.005       # 日内振幅 ≤ 0.5% 视为一字封死板
TURNOVER_COVERAGE = 0.7  # 近端窗口换手率覆盖率达标才用换手率，否则退回成交量
# 疑似吸筹是“完全没有生产形态确认，但原始吸筹分已进入高分尾部”的弱买点。
# 普通股无形态理论上限为40；股东数据缺失但有回购时为35，公司资金面完全缺失时为30。
# ETF 排除公司行为因子后重新归一，无形态理论上限约29.41，因此35分口径下不会产生ETF疑似吸筹点。
SUSPECT_ACCUM_SCORE = 35.0  # 仅 fired 为空且吸筹分达标 → 疑似吸筹(待确认)
SUSPECT_ACCUM_PATTERN_CODE = "SUSPECT_ACCUM"
COMPRESS_AMP_RATIO = 0.96  # P23：近20日振幅 < 此倍×近60日振幅
P23_POSITION_60_MAX = 0.36  # P23：近60日收盘位置分位严格小于36%
P23_MA20_FLOOR = 1.005      # P23：确认状态须收盘至少高于MA20 0.5%
P23_DAY_VOLUME_RATIO_MIN = 0.40  # P23：当日量能/此前20日均量下界（含）
P23_DAY_VOLUME_RATIO_MAX = 1.20  # P23：当日量能/此前20日均量上界（含）
PATTERN_CONFIRM_DAYS = 5   # P24：连续确认后只在首次成立日触发一次
OBV_POSITION_MAX = 0.50
OBV_PRICE_RETURN_MAX = 0.03
OBV_RETURN_MIN = 0.10      # P24：30日量纲化OBV净流入下限
OBV_DIV_MIN = 0.25         # P24：OBV净流入(量纲化)−价格涨幅的最小背离
P2_WINDOW = 10             # P2：近X日反复出现真实长下影（优化研究见纪要25）
P2_MIN_COUNT = 2           # P2：X日内至少Y次，且当日自身必须是其中一次；所有池参数相同
P2_LOWER_SHADOW_MIN = 0.020
P2_SHADOW_BODY_RATIO_MIN = 2.0
P2_DAY_VOLUME_RATIO_MIN = 0.40
P2_DAY_VOLUME_RATIO_MAX = 1.00
P3_POSITION_MAX = 0.32
P3_DROP_MIN = 0.042
P3_EVENT_WINDOW = 10
P3_EVENT_VOLUME_RATIO_MAX = 1.00
P3_UNDERCUT_MAX = 0.06
P3_CONFIRM_VOLUME_RATIO_MAX = 1.35
P5_POSITION_MAX = 0.45
P5_SECOND_LOW_RETURN_MIN = -0.02
P5_SECOND_LOW_RETURN_MAX = 0.10
P5_NECK_REBOUND_MIN = 0.05
P5_SECOND_LOW_RECENCY_MAX = 8
P5_RIGHT_LEG_BREAKOUT_WIN = 3
P5_NECK_RATIO_MIN = 1.00
P5_NECK_RATIO_MAX = 1.03
P5_FLOAT_EPSILON = 1e-12
P15_EVENT_HIGH_DAYS = 60
P15_EVENT_LOOKBACK_DAYS = 15
P15_EVENT_LOW_VOLUME_RATIO_MAX = 1.00
P15_EVENT_HIGH_VOLUME_RATIO_MIN = 1.80
P15_EVENT_RET5_MAX = 0.01
P15_PULLBACK_HIGH_DAYS = 10
P15_PULLBACK_MIN = 0.04
P15_DAY_VOLUME_RATIO_MIN = 1.50
P15_DAY_VOLUME_BASE_DAYS = CHIP_WINNER_RISK_VOLUME_BASE_WIN
P15_DAY_VOLUME_MIN_OBS = CHIP_WINNER_RISK_VOLUME_MIN_OBS
# 当前确认日需要额外保留15根，才能按每个历史事件当日的完整180根窗口
# 复算量源；评分上下文本身仍只取末尾LOOKBACK根，不增加因子维度。
PATTERN_EVAL_BARS = LOOKBACK + max(CHIP_WINNER_RISK_DAYS, P15_EVENT_LOOKBACK_DAYS)
P16_POSITION_MIN = 0.80
P16_MAX_VOLUME_DAYS = 40
P16_EVENT_RECENCY_DAYS = 3
P16_PEAK_HIGH_DAYS = 10
P16_PEAK_DRAWDOWN_MIN = 0.03
P17_POSITION_MIN = 0.80
P17_RECENT_DAYS = 10
P17_BASE_DAYS = 6
P17_RUNUP_MIN = 0.16
P17_PULLBACK_MIN = 0.08
P17_PEAK_RECENCY_BARS = 5
P21_BOX_DAYS = 40
P21_RECLAIM_DAYS = 5
P21_UNDERCUT_RATIO_MAX = 0.975
P21_DAY_VOLUME_RATIO_MAX = 0.75
P20_VOLUME_RATIO_MIN = 1.30
P20_DAY_RETURN_MAX = -0.01
# 与正式 patterns 回测首个锚点完全一致：180 日量度来源窗口 + P26 预留的
# 5 日筹码回看。P21 本身不使用筹码，但统一资格可避免新股线上命中却不在回测样本中。
P21_MIN_HISTORY_BARS = LOOKBACK + CHIP_WINNER_RISK_DAYS
P4_POSITION_MAX = 0.60
P4_VOLUME_RATIO_MIN = 1.20
P4_VOLUME_RATIO_MAX = 2.50
P4_ABS_DRIFT_MAX = 0.06
P4_CMF_MIN = 0.0
P1_POSITION_60_MAX = 0.40
P1_ABS_RET_20_MAX = 0.04
P1_MA20_FLOOR = 0.995
P1_DAY_VOLUME_RATIO_MIN = 0.40
P1_DAY_VOLUME_RATIO_MAX = 1.00
P1_CONFIRM_DAYS = 2
P1_VERSION = "p1_excess_priority_v1"
P1_REPLACED_ON = "2026-07-14"
P1_BREAKOUT_DAYS = 10
P1_ADDON_POSITION_60_MAX = 0.45
P1_ADDON_ABS_RET_20_MAX = 0.06
P1_ADDON_MA20_FLOOR = 0.995
P1_ADDON_DAY_VOLUME_RATIO_MIN = 0.20
P1_ADDON_DAY_VOLUME_RATIO_MAX = 1.20
P1_MARKET_MA20_FLOOR = 0.99
P1_MARKET_MA60_FLOOR = 0.95
P1_MARKET_RET5_MIN = -0.06
P1_MARKET_RET5_MAX = 0.08
P1_MARKET_RET20_MIN = -0.05
P1_MARKET_RET20_MAX = 0.10
P1_MARKET_MA20_SLOPE5_MIN = -0.015
P25_POSITION_120_MAX = 0.65
P25_RANGE_60_MAX = 0.35
P25_ABS_RET_60_MAX = 0.16
P25_VOLUME_CONTRACT_MAX = 0.70
P25_VOLUME_SLOPE_60_MAX = -0.005
P25_RET_20_MIN = 0.005
P25_BREAKOUT_20_MIN = -0.04

# 资金面维度（数据由 stock_crawl_holders.py / stock_crawl_capital.py 入库，见纪要(13)）：
#   股东户数下降/公司回购作为吸筹侧原始特征；龙虎榜上榜进出货侧避雷。
#   v3.3.x 起线上吸筹/出货总分都用原始特征直接加权，不再用截面 rank 做合成。
HOLDER_TABLE = "shareholder_count"     # code, disclose_date, change_pct(户数增减%)
REPURCHASE_TABLE = "repurchase"        # code, disclose_date
LHB_TABLE = "lhb_all"                  # code, date(上榜日)
CAPITAL_EVENT_DAYS = 90                # 回购/上榜：近 N 自然日内有事件视为"近期"
ACCUM_MODEL_WEIGHTS = {
    "chip": 0.1,                        # 低位筹码集中
    "position": 0.1,                    # 价格中低位
    "cmf_eff": 0.1,                     # CMF 反向有效分：高买压反转风险不加分
    "p1": 0.1,                          # P1 两日企稳 OR 首次10日突破+市场门；双池10日有效
    "p2": 0.05,                         # P2 低位真实长下影 + 温和缩量承接
    "p3": 0.1,                          # P3 缩量收盘大跌后首次收复
    "p5": 0.05,                         # P5 双底右腿确认；临时核心有效，小盘更有效
    "p21": 0.1,                         # P21 深破箱底后缩量收回
    "p23": 0.05,                        # P23 低位压缩后温和量能转强；核心有效
    "p24": 0.05,                        # P24 OBV 底背离：连续确认后的首次成立日
    "p25": 0.05,                        # P25 中低位缩量平台价格转强；双池10/20日正向
    "holder_change": 0.1,               # 股东户数变化：户数降=高分，缺失=中性50
    "repurchase": 0.05,                 # 公司回购：近90日回购=100，否则0
}
ACCUM_FEATURES = tuple(ACCUM_MODEL_WEIGHTS.keys())
ETF_ACCUM_EXCLUDED = ("holder_change", "repurchase")
ACCUM_EXPERIMENT_HORIZONS = (5, 10, 20)
ACCUM_EXPERIMENT_GRID_STEP = 0.05
ACCUM_EXPERIMENT_MIN_WEIGHT = 0.05
ACCUM_EXPERIMENT_SPREAD_W = 3.0

# 出货风险分（连续 0~100）：高位 + 阶段涨幅大 + 高位放量 = 派发特征。沿用早期形态研究
# 的「出货分渐变惩罚」——不再只靠形态硬覆盖标签，而是连续地给吸筹分打折，让"既像吸筹又带派发味"的票排名下沉。
DIST_POS_START = 0.70    # 收盘价分位 ≥ 此值才进入派发风险区（高位门控）
DIST_POS_FULL = 0.92     # 分位 ≥ 此值 → 高位满格
DIST_RUNUP_LO = 0.20     # 近20日涨幅 ≥ 此值开始计派发风险
DIST_RUNUP_HI = 0.60     # 涨幅 ≥ 此值 → 满格（已大涨）
DIST_VOL_LO = 1.5        # 量比 ≥ 此值开始计高位放量
DIST_VOL_HI = 3.0        # 量比 ≥ 此值 → 满格（天量见天价）
DIST_MODEL_WEIGHTS = {
    "p14": 0.05,          # 高位放量滞涨
    "p15": 0.05,          # 新高量背离放量回撤确认
    "p16": 0.10,          # 阴天量
    "p17": 0.10,          # 倒V反转
    "p19": 0.10,          # 灌压巨量大阴
    "p20": 0.05,          # 均线放量破位
    "p22": 0.05,          # 放量假突破
    "p26": 0.10,          # 获利盘冲高回撤风险
    "lhb_recent": 0.10,   # 近90日龙虎榜：反向避雷信号
    "technical": 0.15,    # 连续高位派发分：高位 + 20日涨幅 + 高位放量
    "divergence": 0.15,   # 原始 divergence 分，不用 div_eff
}
DIST_FEATURES = tuple(DIST_MODEL_WEIGHTS.keys())
ETF_DIST_EXCLUDED = ("lhb_recent",)
DIST_EXPERIMENT_HORIZONS = (5, 10, 20)
DIST_EXPERIMENT_GRID_STEP = 0.05
DIST_EXPERIMENT_MIN_WEIGHT = 0.1
DIST_EXPERIMENT_SPREAD_W = 3.0
EXPERIMENT_VALIDATION_CANDIDATES = 200
OPPORTUNITY_DISTRIBUTION_PENALTY = 0.5
OPPORTUNITY_FORMULA = "accumulation_percentile * (1 - 0.5 * distribution_percentile / 100)"

# 阶段“把握”中的出货分支：2017-04-14~2026-06-15、1,110只当前池成分、
# 130,464个PIT事件上做时间切分校准（<=2023训练、2024选参、2025+留出测试）。
# 形态代码负责触发出货阶段；样本外结果显示形态身份/数量的可靠度不稳定，因此把握只用
# 同日可见的吸筹分A与连续出货分D做保守Logistic校准。完整实验见
# data/capital/hot_money_phase_confidence_experiment.json。
PHASE_CONFIDENCE_SELL_MODEL_VERSION = "sell_logit_20260715_v1"
PHASE_CONFIDENCE_SELL_INTERCEPT = 0.04397102
PHASE_CONFIDENCE_SELL_ACCUMULATION_COEF = -0.60109165
PHASE_CONFIDENCE_SELL_DISTRIBUTION_COEF = 1.10926961

# 阶段标签按游资操作顺序排列（疑似吸筹→吸筹→试盘→洗盘→突破→拉升→出货，观望=场外）。
# 表头计数据此从左到右展示；标签字符串须与 _pattern_phase() 的返回值完全一致。
PHASE_ORDER: Tuple[str, ...] = (
    "疑似吸筹(待确认)🟢",
    "吸筹🟢",
    "试盘🟡",
    "洗盘🟡",
    "吸筹+洗盘🟡",
    "▲突破🟠",
    "拉升中🟠",
    "出货预警🔴",
    "观望⚪",
)

# 形态回测的生产买点：命中任一代码即标记买入。该集合与有效性元数据
# 分开维护，避免回测标记规则被前端展示逻辑隐式改写。
PATTERN_BACKTEST_BUY = frozenset({"P1", "P2", "P3", "P5", "P21", "P23", "P24", "P25"})

# 出货预警：下列任一形态命中即触发。按代码去重后每项记1分；保留两个阈值
# 常量和原 metadata 字段，兼容既有列表、回测与前端消费者。
DISTRIBUTION_WARNING_PATTERN_CODES = frozenset({
    "P14", "P15", "P16", "P17", "P19", "P20", "P22", "P26",
})
DISTRIBUTION_WARNING_POINTS = {
    code: 1 for code in sorted(DISTRIBUTION_WARNING_PATTERN_CODES)
}
DISTRIBUTION_WARNING_EFFECTIVE_SELL_THRESHOLD = 1
DISTRIBUTION_WARNING_POINTS_THRESHOLD = 1
# 市值上限(亿)：传给 --exclude-large-cap 时生效。默认「不剔除/全市值」(用户选择)，
# 仅在显式 --exclude-large-cap 时启用，用于按市值分层查看小/中盘子集。
MAX_MARKET_CAP_YI = 300.0

# 右进左出触发器：吸筹分给"状态"，触发器给"买点"——从低/中位整理放量突破近端高点。
TRIGGER_WIN = 20          # 突破窗口：放量突破近 20 日收盘高点
TRIGGER_VOL_MULT = 1.3    # 突破当日量 > 1.3× 前 20 日均量
TRIGGER_BASE_MAX_POS = 0.70  # 被突破的前高在近 60 日分位 < 0.70 → 从低/中位突破(非高位追涨)

# ── 后验回测参数 ──────────────────────────────────────────────
VERIFY_HORIZONS = (2, 5, 10, 20, 40)   # 游资持有期（交易日）：2/5/10/20/40 日前向收益，逐周期判有效
VERIFY_STEP = 1                    # 逐交易日滚动；事件研究无需训练/拟合，不稀疏抽样
VERIFY_WINDOW_DAYS = 0             # 回测窗口（交易日）；0 = 使用数据库全部可用历史
PATTERN_FDR_Q = 0.10               # 每个股票池内对全部形态×周期做 BH-FDR
VERIFY_MIN_NAMES = 30              # 单个截面至少多少只票才计入截面 IC / 多空
VERIFY_BUCKETS = 5                 # 吸筹分分位桶数（五等分）


# ── 基础工具 ──────────────────────────────────────────────────

def _limit_pct(code: str) -> float:
    """个股涨跌停幅度（粗分；ST 5% 细节忽略，作为噪声接受）。"""
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    if code.startswith(("8", "4")):  # 北交所
        return 30.0
    return 10.0


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _verify_window(values: Sequence[str]) -> List[str]:
    """按后验研究窗口截取日期；VERIFY_WINDOW_DAYS<=0 表示保留全部可用历史。"""
    dates = list(values)
    if VERIFY_WINDOW_DAYS <= 0:
        return dates
    return dates[-VERIFY_WINDOW_DAYS:]


def _verify_as_of_dates(values: Sequence[str]) -> List[str]:
    """从窗口末端锚定抽样网格，避免改变窗口长度时平移近端 as-of 日期。"""
    dates = _verify_window(values)
    if not dates:
        return []
    offset = len(dates) % VERIFY_STEP
    return dates[offset::VERIFY_STEP]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _ranks(values: Sequence[float]) -> List[float]:
    """平均秩（处理并列），供 Spearman 用。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 5:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _clip01(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else value


def _safe(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_stock_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else ""


def _eastmoney_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if not text or text in {"-", "--", "—", "None", "nan", "加载中..."}:
        return None
    return _safe(text)


def _eastmoney_quote_time(value: Any) -> Optional[str]:
    ts = _eastmoney_number(value)
    if not ts or ts <= 0:
        return None
    if ts > 100000000000:
        ts = ts / 1000.0
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return None


def _market_symbol(code: str) -> str:
    code = _clean_stock_code(code)
    if code.startswith(("4", "8", "92")):
        return f"bj{code}"
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _parse_compact_time(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if len(text) < 14 or not text[:14].isdigit():
        return None
    return f"{text[:4]}-{text[4:6]}-{text[6:8]} {text[8:10]}:{text[10:12]}:{text[12:14]}"


def _quote_chunks(codes: Sequence[str], size: int = 180) -> List[List[str]]:
    clean = [_clean_stock_code(code) for code in codes]
    clean = [code for code in dict.fromkeys(clean) if code]
    return [clean[i:i + size] for i in range(0, len(clean), size)]


TRADING_MINUTES_PER_DAY = 240.0
INTRADAY_VOLUME_MIN_ELAPSED = 20.0
INTRADAY_VOLUME_FACTOR_CAP = 6.0
INTRADAY_U_SHAPE_VOLUME_CURVE: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (5.0, 0.045),
    (15.0, 0.12),
    (30.0, 0.20),
    (60.0, 0.33),
    (90.0, 0.43),
    (120.0, 0.50),
    (150.0, 0.58),
    (180.0, 0.68),
    (210.0, 0.82),
    (230.0, 0.94),
    (240.0, 1.0),
)


def _parse_quote_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt, size in (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%d %H:%M", 16),
        ("%Y%m%d%H%M%S", 14),
    ):
        try:
            return datetime.strptime(text[:size], fmt)
        except ValueError:
            continue
    return None


def _a_share_elapsed_minutes(ts: datetime) -> float:
    minute = ts.hour * 60.0 + ts.minute + ts.second / 60.0
    morning_start = 9 * 60 + 30
    morning_end = 11 * 60 + 30
    afternoon_start = 13 * 60
    afternoon_end = 15 * 60
    if minute <= morning_start:
        return 0.0
    if minute <= morning_end:
        return max(0.0, minute - morning_start)
    if minute <= afternoon_start:
        return 120.0
    if minute <= afternoon_end:
        return 120.0 + minute - afternoon_start
    return TRADING_MINUTES_PER_DAY


def _u_shape_volume_cumulative_share(elapsed_minutes: float) -> float:
    elapsed = max(0.0, min(TRADING_MINUTES_PER_DAY, elapsed_minutes))
    points = INTRADAY_U_SHAPE_VOLUME_CURVE
    if elapsed <= points[0][0]:
        return points[0][1]
    for (left_min, left_share), (right_min, right_share) in zip(points, points[1:]):
        if elapsed <= right_min:
            span = right_min - left_min
            if span <= 0:
                return right_share
            weight = (elapsed - left_min) / span
            return left_share + (right_share - left_share) * weight
    return points[-1][1]


def _intraday_volume_projection(
    quote: Dict[str, Any], *, now: Optional[datetime] = None
) -> Tuple[float, Optional[float], bool]:
    """盘中累计量不是全天量：按内置 U 型日内成交曲线折算，收盘后保持原值。"""
    now = now or datetime.now()
    quote_dt = _parse_quote_datetime(quote.get("quote_time"))
    if quote_dt is None:
        quote_date = str(quote.get("quote_date") or "")[:10]
        if quote_date == now.strftime("%Y-%m-%d"):
            quote_dt = datetime.strptime(f"{quote_date} {now:%H:%M:%S}", "%Y-%m-%d %H:%M:%S")
    if quote_dt is None:
        return 1.0, None, False
    # A cached quote from a previous trading day is already a completed daily
    # observation.  Projecting its morning timestamp as if it were live can
    # multiply volume/turnover several times over.
    if quote_dt.date() != now.date():
        return 1.0, _a_share_elapsed_minutes(quote_dt), False
    elapsed = _a_share_elapsed_minutes(quote_dt)
    if elapsed <= 0.0 or elapsed >= TRADING_MINUTES_PER_DAY:
        return 1.0, elapsed, False
    effective_elapsed = max(elapsed, INTRADAY_VOLUME_MIN_ELAPSED)
    cumulative_share = _u_shape_volume_cumulative_share(effective_elapsed)
    if cumulative_share <= 0.0:
        return 1.0, elapsed, False
    factor = min(INTRADAY_VOLUME_FACTOR_CAP, 1.0 / cumulative_share)
    return factor, elapsed, factor > 1.0001


def _parse_tencent_quote_text(text: str) -> Dict[str, Dict[str, Any]]:
    quotes: Dict[str, Dict[str, Any]] = {}
    for symbol, body in re.findall(r'v_([a-z]{2}\d{6})="([^"]*)"', text or ""):
        parts = body.split("~")
        if len(parts) < 35:
            continue
        code = _clean_stock_code(parts[2] if len(parts) > 2 else symbol)
        if not code:
            continue
        quote_time = _parse_compact_time(parts[30] if len(parts) > 30 else None)
        volume = None
        amount = None
        if len(parts) > 35 and "/" in parts[35]:
            deal = parts[35].split("/")
            if len(deal) >= 3:
                volume = _eastmoney_number(deal[1])
                amount = _eastmoney_number(deal[2])
        if volume is None and len(parts) > 36:
            volume = _eastmoney_number(parts[36])
        if amount is None and len(parts) > 37:
            amount_wan = _eastmoney_number(parts[37])
            amount = amount_wan * 10000.0 if amount_wan is not None else None
        market_cap_yi = None
        if len(parts) > 45:
            market_cap_yi = _eastmoney_number(parts[45])
        if market_cap_yi is None and len(parts) > 44:
            market_cap_yi = _eastmoney_number(parts[44])
        quotes[code] = {
            "code": code,
            "name": parts[1] if len(parts) > 1 else "",
            "price": _eastmoney_number(parts[3] if len(parts) > 3 else None),
            "pre_close": _eastmoney_number(parts[4] if len(parts) > 4 else None),
            "open": _eastmoney_number(parts[5] if len(parts) > 5 else None),
            "volume": volume,
            "amount": amount,
            "change_pct": _eastmoney_number(parts[32] if len(parts) > 32 else None),
            "high": _eastmoney_number(parts[33] if len(parts) > 33 else None),
            "low": _eastmoney_number(parts[34] if len(parts) > 34 else None),
            "turnover": _eastmoney_number(parts[38] if len(parts) > 38 else None),
            "market_cap_yi": market_cap_yi,
            "quote_time": quote_time,
            "quote_date": quote_time[:10] if quote_time else datetime.now().strftime("%Y-%m-%d"),
            "source": "tencent_batch",
        }
    return quotes


def fetch_tencent_a_quotes(codes: Sequence[str], timeout: int = 8) -> Dict[str, Dict[str, Any]]:
    """腾讯批量实时行情。按当前雷达股票池分块请求，避免逐股访问。"""
    quotes: Dict[str, Dict[str, Any]] = {}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    for chunk in _quote_chunks(codes):
        symbols = ",".join(_market_symbol(code) for code in chunk)
        response = requests.get("https://qt.gtimg.cn/q=" + symbols, headers=headers, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.encoding or "GBK"
        quotes.update(_parse_tencent_quote_text(response.text))
    return quotes


def _parse_sina_quote_text(text: str) -> Dict[str, Dict[str, Any]]:
    quotes: Dict[str, Dict[str, Any]] = {}
    for symbol, body in re.findall(r'var hq_str_([a-z]{2}\d{6})="([^"]*)"', text or ""):
        parts = body.split(",")
        code = _clean_stock_code(symbol)
        if len(parts) < 32 or not code or not parts[0]:
            continue
        price = _eastmoney_number(parts[3])
        pre_close = _eastmoney_number(parts[2])
        change_pct = ((price / pre_close - 1.0) * 100.0) if price is not None and pre_close else None
        volume_shares = _eastmoney_number(parts[8])
        quote_date = str(parts[30] or "").strip()
        quote_time = f"{quote_date} {str(parts[31] or '').strip()}" if quote_date and len(parts) > 31 else None
        quotes[code] = {
            "code": code,
            "name": parts[0],
            "price": price,
            "pre_close": pre_close,
            "open": _eastmoney_number(parts[1]),
            "high": _eastmoney_number(parts[4]),
            "low": _eastmoney_number(parts[5]),
            "volume": round(volume_shares / 100.0, 2) if volume_shares is not None else None,
            "amount": _eastmoney_number(parts[9]),
            "change_pct": change_pct,
            "turnover": None,
            "market_cap_yi": None,
            "quote_time": quote_time,
            "quote_date": quote_date or datetime.now().strftime("%Y-%m-%d"),
            "source": "sina_batch",
        }
    return quotes


def fetch_sina_a_quotes(codes: Sequence[str], timeout: int = 8) -> Dict[str, Dict[str, Any]]:
    """新浪批量实时行情。字段少于腾讯，但可作为低成本备用源。"""
    quotes: Dict[str, Dict[str, Any]] = {}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
    for chunk in _quote_chunks(codes):
        symbols = ",".join(_market_symbol(code) for code in chunk)
        response = requests.get("https://hq.sinajs.cn/list=" + symbols, headers=headers, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.encoding or "GB18030"
        quotes.update(_parse_sina_quote_text(response.text))
    return quotes


def fetch_eastmoney_a_spot_quotes(timeout: int = 8) -> Dict[str, Dict[str, Any]]:
    """东方财富全 A 快照。一次请求全市场，供雷达实时刷新本地过滤。"""
    from stock_crawl_segment_leaders import EASTMONEY_A_SPOT_URL, EASTMONEY_HEADERS

    response = requests.get(
        EASTMONEY_A_SPOT_URL,
        params={
            "pn": "1",
            "pz": "10000",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f2,f3,f5,f6,f8,f12,f14,f15,f16,f17,f18,f20,f124",
        },
        headers=EASTMONEY_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    rows = ((response.json().get("data") or {}).get("diff") or [])
    if not isinstance(rows, list):
        raise RuntimeError("东方财富全A实时行情返回结构异常")

    quotes: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _clean_stock_code(row.get("f12"))
        if not code:
            continue
        quote_time = _eastmoney_quote_time(row.get("f124"))
        market_cap_yuan = _eastmoney_number(row.get("f20"))
        quotes[code] = {
            "code": code,
            "name": str(row.get("f14") or "").strip(),
            "price": _eastmoney_number(row.get("f2")),
            "change_pct": _eastmoney_number(row.get("f3")),
            "volume": _eastmoney_number(row.get("f5")),
            "amount": _eastmoney_number(row.get("f6")),
            "turnover": _eastmoney_number(row.get("f8")),
            "high": _eastmoney_number(row.get("f15")),
            "low": _eastmoney_number(row.get("f16")),
            "open": _eastmoney_number(row.get("f17")),
            "pre_close": _eastmoney_number(row.get("f18")),
            "market_cap_yi": round(market_cap_yuan / 1e8, 4) if market_cap_yuan else None,
            "quote_time": quote_time,
            "quote_date": quote_time[:10] if quote_time else datetime.now().strftime("%Y-%m-%d"),
            "source": "eastmoney_a_spot",
        }
    return quotes


def fetch_eastmoney_etf_spot_quotes(timeout: int = 8) -> Dict[str, Dict[str, Any]]:
    """东方财富 ETF 快照兜底；基金规模不冒充公司总市值。"""
    response = requests.get(
        "https://88.push2.eastmoney.com/api/qt/clist/get",
        params={
            "pn": "1", "pz": "10000", "po": "1", "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": "2", "invt": "2",
            "fid": "f12", "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
            "fields": "f2,f3,f5,f6,f8,f12,f14,f15,f16,f17,f18,f21,f124",
        },
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        timeout=timeout,
    )
    response.raise_for_status()
    rows = ((response.json().get("data") or {}).get("diff") or [])
    quotes: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _clean_stock_code(row.get("f12"))
        if not code:
            continue
        quote_time = _eastmoney_quote_time(row.get("f124"))
        fund_scale = _eastmoney_number(row.get("f21"))
        quotes[code] = {
            "code": code,
            "name": str(row.get("f14") or "").strip(),
            "price": _eastmoney_number(row.get("f2")),
            "change_pct": _eastmoney_number(row.get("f3")),
            "volume": _eastmoney_number(row.get("f5")),
            "amount": _eastmoney_number(row.get("f6")),
            "turnover": _eastmoney_number(row.get("f8")),
            "high": _eastmoney_number(row.get("f15")),
            "low": _eastmoney_number(row.get("f16")),
            "open": _eastmoney_number(row.get("f17")),
            "pre_close": _eastmoney_number(row.get("f18")),
            "market_cap_yi": None,
            "fund_scale_yi": round(fund_scale / 1e8, 4) if fund_scale else None,
            "quote_time": quote_time,
            "quote_date": quote_time[:10] if quote_time else datetime.now().strftime("%Y-%m-%d"),
            "source": "eastmoney_etf_spot",
        }
    return quotes


def fetch_realtime_a_quotes(codes: Sequence[str], timeout: int = 8) -> Dict[str, Dict[str, Any]]:
    """当前雷达实时行情入口：腾讯批量优先，新浪备用，东财全 A 兜底。"""
    clean_codes = [_clean_stock_code(code) for code in codes]
    clean_codes = [code for code in dict.fromkeys(clean_codes) if code]
    errors = []
    for label, fetcher in (
        ("tencent_batch", fetch_tencent_a_quotes),
        ("sina_batch", fetch_sina_a_quotes),
    ):
        try:
            quotes = fetcher(clean_codes, timeout=timeout)
            if quotes:
                return quotes
            errors.append(f"{label}: empty")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    if any(code.startswith(("1", "5")) for code in clean_codes):
        try:
            all_etfs = fetch_eastmoney_etf_spot_quotes(timeout=timeout)
            quotes = {code: all_etfs[code] for code in clean_codes if code in all_etfs}
            if quotes:
                return quotes
            errors.append("eastmoney_etf_spot: empty")
        except Exception as exc:
            errors.append(f"eastmoney_etf_spot: {exc}")
    try:
        all_quotes = fetch_eastmoney_a_spot_quotes(timeout=timeout)
        quotes = {code: all_quotes[code] for code in clean_codes if code in all_quotes}
        if quotes:
            return quotes
        errors.append("eastmoney_a_spot: empty")
    except Exception as exc:
        errors.append(f"eastmoney_a_spot: {exc}")
    raise RuntimeError("实时行情源均失败：" + " | ".join(errors))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ── 候选池：细分龙头 / 游资小盘 / ETF 配置池 ──

POOLS = ("leader", "hotmoney", "etf")
DEFAULT_POOL = "leader"


def load_candidates(conn: sqlite3.Connection, pool: str = DEFAULT_POOL,
                    max_cap: Optional[float] = MAX_MARKET_CAP_YI) -> List[Dict[str, Any]]:
    """取候选池：pool='leader' 细分龙头(is_leader) / 'hotmoney' 游资小盘universe(is_hot_money，
    由 stock_crawl_hot_money_universe.py 建池)。

    max_cap>0 时剔除总市值超过该值(亿元)的股票(按市值分层查看小/中盘子集用)；
    市值缺失会先由 stock_storage 从 stock_history 最新非空 market_cap 回补。max_cap=None/0 不过滤。
    """
    if pool == "etf":
        configured = load_etf_pool()
        codes = [row["code"] for row in configured]
        validated_codes: Optional[set[str]] = None
        try:
            validation_report = json.loads(ETF_REFRESH_REPORT_FILE.read_text(encoding="utf-8"))
            report_codes = validation_report.get("validated_codes")
            if (
                validation_report.get("status") in {"ok", "partial"}
                and isinstance(report_codes, list)
            ):
                validated_codes = {
                    str(code).zfill(6) for code in report_codes
                    if str(code).isdigit()
                }
        except (OSError, ValueError, TypeError):
            pass
        meta: Dict[str, sqlite3.Row] = {}
        for start in range(0, len(codes), 500):
            chunk = codes[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT m.code, m.name, m.instrument_type, "
                f"EXISTS(SELECT 1 FROM stock_history h WHERE h.code = m.code) AS has_history "
                f"FROM stock_meta m WHERE m.code IN ({placeholders})",
                chunk,
            ).fetchall():
                meta[str(row["code"]).zfill(6)] = row
        out: List[Dict[str, Any]] = []
        for item in configured:
            code = item["code"]
            stored = meta.get(code)
            # 只分析已经过 ETF 刷新链验证并成功入库行情的代码。新改配置需先点
            # “刷新数据”，避免 LOF/普通股票仅凭六位代码混入 ETF 候选池。
            if (
                (validated_codes is not None and code not in validated_codes)
                or
                stored is None
                or str(stored["instrument_type"] or "stock") != "etf"
                or not bool(stored["has_history"])
            ):
                continue
            category = str(item.get("category") or "ETF")
            out.append({
                "code": code,
                "name": str((stored and stored["name"]) or item.get("name") or f"ETF {code}"),
                "market_cap_yi": None,
                "segment_name": category,
                "parent_segment": category,
                "asset_type": "etf",
            })
        return out

    rows = stock_storage.pool_members(conn, pool)
    out: List[Dict[str, Any]] = []
    for row in rows:
        code = stock_storage._normalize_code(row.get("code"))
        if not code:
            continue
        cap = row.get("market_cap_yi")
        if max_cap and cap is not None and cap > max_cap:
            continue
        out.append({
            "code": code,
            "name": str(row.get("name") or ""),
            "market_cap_yi": cap,
            "segment_name": row.get("segment_name") or "",
            "parent_segment": row.get("parent_segment") or "",
            "asset_type": "stock",
        })
    return out


def load_leader_candidates(conn: sqlite3.Connection,
                           max_cap: Optional[float] = MAX_MARKET_CAP_YI) -> List[Dict[str, Any]]:
    """细分龙头候选——load_candidates(pool='leader') 的兼容别名。"""
    return load_candidates(conn, DEFAULT_POOL, max_cap)


def _bar(row: sqlite3.Row) -> Dict[str, Any]:
    """把一行日线整成轻量短键 dict（解耦 DB 列名、PIT 切片更快）。"""
    bar = {
        "date": row["date"],
        "open": _safe(row["daily_open"]),
        "high": _safe(row["daily_high"]),
        "low": _safe(row["daily_low"]),
        "close": _safe(row["daily_close"]),
        "volume": _safe(row["daily_volume"]),
        "amount": _safe(row["daily_amount"]),
        "chg": _safe(row["daily_change_pct"]),
        "turnover": _safe(row["daily_turnover_rate"]),
    }
    # 历史回测的相邻窗口会重复访问同一根 bar 数百次；均价只依赖该 bar，
    # 入库读取时算一次即可，避免在每个筹码窗口里重复做成交额/成交量换算。
    bar["_chip_avg_price"] = _bar_avg_price(bar)
    return bar


_BAR_SQL = (
    "SELECT date, daily_open, daily_high, daily_low, daily_close, daily_volume, daily_amount, "
    "daily_change_pct, daily_turnover_rate FROM stock_history "
    "WHERE code = ? AND daily_close IS NOT NULL AND daily_volume IS NOT NULL "
)


def _recent_bars(conn: sqlite3.Connection, code: str, limit: int = LOOKBACK,
                 as_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """该 code 近 limit 个有效日线 bar（升序），跳过估值快照空行。

    as_of 给定时只取该日期及以前的 bar（PIT 防泄漏，供历史 as-of 复盘）。
    """
    if as_of:
        rows = conn.execute(_BAR_SQL + "AND date <= ? ORDER BY date DESC LIMIT ?",
                            (code, as_of, limit)).fetchall()
    else:
        rows = conn.execute(_BAR_SQL + "ORDER BY date DESC LIMIT ?", (code, limit)).fetchall()
    return [_bar(r) for r in reversed(rows)]


def _bulk_recent_bars(conn: sqlite3.Connection, codes: Sequence[str],
                      limit: int = LOOKBACK) -> Dict[str, List[Dict[str, Any]]]:
    """批量取多只股票近端有效日线，避免实时刷新时对 1000 只股票逐只查库。"""
    clean_codes = [str(c).zfill(6) for c in dict.fromkeys(codes) if c]
    out: Dict[str, List[Dict[str, Any]]] = {code: [] for code in clean_codes}
    if not clean_codes:
        return out
    select_cols = (
        "code, date, daily_open, daily_high, daily_low, daily_close, daily_volume, "
        "daily_amount, daily_change_pct, daily_turnover_rate"
    )
    try:
        for i in range(0, len(clean_codes), 800):
            chunk = clean_codes[i:i + 800]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT {select_cols} FROM ("
                f"SELECT {select_cols}, "
                "ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn "
                "FROM stock_history "
                f"WHERE code IN ({placeholders}) "
                "AND daily_close IS NOT NULL AND daily_volume IS NOT NULL"
                ") WHERE rn <= ? ORDER BY code, date",
                (*chunk, limit),
            ).fetchall()
            for row in rows:
                out.setdefault(row["code"], []).append(_bar(row))
    except sqlite3.OperationalError:
        # 极旧 SQLite 无 window function 时降级，仍保持功能可用。
        return {code: _recent_bars(conn, code, limit=limit) for code in clean_codes}
    return out


def _all_bars(conn: sqlite3.Connection, code: str) -> List[Dict[str, Any]]:
    """该 code 全部有效日线 bar（升序）。供 verify 做 PIT 滑窗。"""
    rows = conn.execute(_BAR_SQL + "ORDER BY date", (code,)).fetchall()
    return [_bar(r) for r in rows]


# ── 潜伏分（纯函数，ambush 取最新窗口 / verify 取历史窗口都复用）──

def _volume_series(bars: List[Dict[str, Any]]) -> Tuple[List[Optional[float]], str]:
    """挑量度量：近端换手率覆盖率达标用换手率（更干净，qfq 不复权 volume 有跳变），否则退回成交量。"""
    turns = [b["turnover"] for b in bars]
    coverage = sum(1 for t in turns if t is not None) / len(turns) if turns else 0.0
    if coverage >= TURNOVER_COVERAGE:
        return turns, "turnover"
    return [b["volume"] for b in bars], "volume"


def _score_volume_ratio(vol: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """F1 量比抬升：近 SHORT_WIN 日均量 / 紧邻其前 BASE_WIN 日均量。"""
    if len(vol) < SHORT_WIN + BASE_WIN:
        return None, None
    short = [v for v in vol[-SHORT_WIN:] if v is not None]
    base = [v for v in vol[-(SHORT_WIN + BASE_WIN):-SHORT_WIN] if v is not None]
    short_avg, base_avg = _mean(short), _mean(base)
    if not short_avg or not base_avg or base_avg <= 0:
        return None, None
    ratio = short_avg / base_avg
    score = _clip01((ratio - 1.0) / (VOL_RATIO_FULL - 1.0)) * 100.0
    return score, ratio


def _p26_volume_ratio(vol: Sequence[Optional[float]]) -> Optional[float]:
    """P26 专用量能确认：当日量能 / 此前20日均量；不使用通用的近5日均量口径。"""
    required = CHIP_WINNER_RISK_VOLUME_BASE_WIN + 1
    if len(vol) < required:
        return None
    try:
        current = float(vol[-1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current) or current <= 0:
        return None
    prior: List[float] = []
    for value in vol[-required:-1]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric > 0:
            prior.append(numeric)
    if len(prior) < CHIP_WINNER_RISK_VOLUME_MIN_OBS:
        return None
    base = _mean(prior)
    return current / base if base and base > 0 else None


def _score_position(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """A2 价格中低位：收盘价在近 HIGH_WIN 日分位越低越好（留出上行空间，已拉升=0）。"""
    closes = [b["close"] for b in bars[-HIGH_WIN:] if b["close"] is not None]
    if len(closes) < 20:
        return None, None
    last = closes[-1]
    pct = sum(1 for c in closes if c <= last) / len(closes)
    score = _clip01((POS_HIGH - pct) / (POS_HIGH - POS_LOW)) * 100.0
    return score, pct


def _amp_ratio(bars: List[Dict[str, Any]]) -> Optional[float]:
    """近20日振幅均值 / 近60日振幅均值（<1 = 波动压缩/酝酿）。振幅=(高−低)/昨收。样本<60 返回 None。"""
    amps: List[float] = []
    for i in range(1, len(bars)):
        pc, h, l = bars[i - 1]["close"], bars[i]["high"], bars[i]["low"]
        if pc and h is not None and l is not None:
            amps.append((h - l) / pc)
    if len(amps) < 60:
        return None
    a_recent, a_base = _mean(amps[-20:]), _mean(amps[-60:])
    return (a_recent / a_base) if a_base else None


def _turnover_pctile(bars: List[Dict[str, Any]]) -> Optional[float]:
    """最新换手率在近端窗口内的分位（0~1）：越高=换手越拥挤。样本<20 返回 None。"""
    turns = [b["turnover"] for b in bars if b["turnover"] is not None]
    if len(turns) < 20:
        return None
    last = turns[-1]
    return sum(1 for t in turns if t <= last) / len(turns)


def _absorption_score(drift: float) -> float:
    """20 日累计涨跌幅 → 吸筹分。横盘微涨最佳；大跌(杀跌/出货)或大涨(已拉升)都归零。"""
    if drift <= -0.12 or drift >= 0.25:
        return 0.0
    if drift < -0.03:
        return (drift + 0.12) / 0.09 * 100.0     # -12%→0 升到 -3%→100
    if drift <= 0.08:
        return 100.0                              # -3%~+8% 横盘微涨：满分
    return (0.25 - drift) / 0.17 * 100.0          # +8%→100 降到 +25%→0


def _score_absorption(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """近 DRIFT_WIN 日价格横住程度（结果端），杀跌/暴涨都不算吸筹。"""
    closes = [b["close"] for b in bars if b["close"] is not None]
    if len(closes) < DRIFT_WIN + 1:
        return None, None
    ref = closes[-1 - DRIFT_WIN]
    if not ref:
        return None, None
    drift = closes[-1] / ref - 1.0
    return _absorption_score(drift), drift


def _score_divergence(bars: List[Dict[str, Any]], vol: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """信号①努力与结果背离：放量(努力) × 价格横住(结果) 连乘。放量却不涨=有人在吸货。"""
    s_vol, vol_ratio = _score_volume_ratio(vol)
    s_abs, drift = _score_absorption(bars)
    if vol_ratio is None or s_abs is None:
        return None, drift
    effort = _clip01((vol_ratio - 1.0) / (VOL_RATIO_FULL - 1.0))   # 量能抬升强度 0~1
    return effort * (s_abs / 100.0) * 100.0, drift


def _money_flow_mult(bar: Dict[str, Any]) -> Optional[float]:
    """Chaikin 资金流乘数 MFM = ((C−L)−(H−C))/(H−L) ∈ [−1,1]，收越靠上半部越正。"""
    h, l, c = bar["high"], bar["low"], bar["close"]
    if h is None or l is None or c is None or h <= l:
        return None   # 一字板 high==low → 无定义，跳过
    return ((c - l) - (h - c)) / (h - l)


def _score_cmf(bars: List[Dict[str, Any]], vol: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """信号②收盘买压：Chaikin Money Flow = Σ(MFM×量)/Σ量，收盘持续靠上半部=买方吸收。"""
    num = den = 0.0
    for b, v in zip(bars[-CMF_WIN:], vol[-CMF_WIN:]):
        m = _money_flow_mult(b)
        if m is None or v is None:
            continue
        num += m * v
        den += v
    if den <= 0:
        return None, None
    cmf = num / den   # [-1, 1]
    return _clip01(cmf / CMF_FULL) * 100.0, cmf


def _bar_avg_price(bar: Dict[str, Any]) -> Optional[float]:
    """用成交额/成交量估算均价；不可用时退回典型价，供三角筹码分布作为峰值。"""
    if "_chip_avg_price" in bar:
        return bar["_chip_avg_price"]
    h, l, c = bar["high"], bar["low"], bar["close"]
    amount, volume = bar.get("amount"), bar.get("volume")
    if amount and volume and amount > 0 and volume > 0 and h is not None and l is not None:
        for avg in (amount / (volume * 100.0), amount / volume):
            if l <= avg <= h:
                return avg
    vals = [x for x in (h, l, c) if x is not None]
    return sum(vals) / len(vals) if vals else None


def _triangular_weights(prices: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    """当日换手筹码在 low-peak-high 间做三角分布。"""
    if len(prices) == 1 or high <= low:
        return np.ones(len(prices))
    peak = min(high, max(low, peak))
    weights = np.zeros(len(prices))
    if peak <= low:
        weights = (high - prices) / (high - low)
    elif peak >= high:
        weights = (prices - low) / (high - low)
    else:
        left = prices <= peak
        weights[left] = (prices[left] - low) / (peak - low)
        weights[~left] = (high - prices[~left]) / (high - peak)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights[np.argmin(np.abs(prices - peak))] = 1.0
    return weights


def _chip_metrics(bars: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """信号③成本分布：换手衰减 + 当日筹码在 low-avg-high 三角铺开，重建流通筹码成本分布。

    返回 concentration(主峰±CHIP_BAND 内筹码占比)、peak_pctile(主峰价位分位)、
    price_to_peak(当前价相对主峰)、winner(获利盘比例)。换手率缺失太多则返回 None。
    """
    lows, highs, peaks, turns = [], [], [], []
    for b in bars:
        avg = _bar_avg_price(b)
        if b["low"] and b["high"] and avg is not None and b["turnover"] is not None and b["high"] > b["low"]:
            lows.append(b["low"]); highs.append(b["high"]); peaks.append(avg); turns.append(b["turnover"])
    if len(lows) < 30:
        return None
    pmin, pmax = min(lows), max(highs)
    if pmax <= pmin:
        return None
    grid = np.linspace(pmin, pmax, CHIP_BUCKETS)
    span = pmax - pmin
    low_array = np.asarray(lows, dtype=float)
    high_array = np.asarray(highs, dtype=float)
    peak_array = np.clip(np.asarray(peaks, dtype=float), low_array, high_array)
    frac_array = np.clip(
        np.asarray(turns, dtype=float) / 100.0 * CHIP_DECAY,
        0.0,
        1.0,
    )

    # 旧实现逐日创建一个很短的 NumPy 数组，180 日窗口会触发约 180 次
    # _triangular_weights 调用。这里一次广播生成所有日的三角分布；最终筹码递推
    #   chips_t = chips_(t-1) * (1-frac_t) + frac_t * day_dist_t
    # 等价为各日分布按“当日换手 × 后续存活率”加权求和。
    prices = grid[None, :]
    lo = low_array[:, None]
    hi = high_array[:, None]
    peak = peak_array[:, None]
    width = hi - lo
    left_width = np.where(peak > lo, peak - lo, 1.0)
    right_width = np.where(peak < hi, hi - peak, 1.0)
    regular = np.where(
        prices <= peak,
        (prices - lo) / left_width,
        (hi - prices) / right_width,
    )
    weights = np.where(
        peak <= lo,
        (hi - prices) / width,
        np.where(peak >= hi, (prices - lo) / width, regular),
    )

    i0 = np.floor((low_array - pmin) / span * (CHIP_BUCKETS - 1)).astype(int)
    i1 = np.floor((high_array - pmin) / span * (CHIP_BUCKETS - 1)).astype(int)
    i0 = np.clip(i0, 0, CHIP_BUCKETS - 1)
    i1 = np.clip(np.maximum(i0, i1), 0, CHIP_BUCKETS - 1)
    bucket_ids = np.arange(CHIP_BUCKETS)[None, :]
    weights = np.maximum(weights, 0.0)
    weights[(bucket_ids < i0[:, None]) | (bucket_ids > i1[:, None])] = 0.0
    row_sums = weights.sum(axis=1)
    empty_rows = np.flatnonzero(row_sums <= 0.0)
    if len(empty_rows):
        empty_bucket_ids = bucket_ids < i0[empty_rows, None]
        empty_bucket_ids |= bucket_ids > i1[empty_rows, None]
        distances = np.abs(grid[None, :] - peak_array[empty_rows, None])
        distances[empty_bucket_ids] = np.inf
        nearest = distances.argmin(axis=1)
        weights[empty_rows, nearest] = 1.0
        row_sums[empty_rows] = 1.0
    weights /= row_sums[:, None]

    coefficients = np.empty(len(frac_array), dtype=float)
    survival = 1.0
    for index in range(len(frac_array) - 1, -1, -1):
        coefficients[index] = frac_array[index] * survival
        survival *= 1.0 - frac_array[index]
    chips = coefficients @ weights
    tot = chips.sum()
    if tot <= 0:
        return None
    chips /= tot
    close = bars[-1]["close"]
    peak_price = float(grid[int(chips.argmax())])
    concentration = float(chips[np.abs(grid - peak_price) <= CHIP_BAND * peak_price].sum())
    winner = float(chips[grid <= close].sum()) if close else 0.0
    peak_pctile = (peak_price - pmin) / span
    price_to_peak = close / peak_price - 1.0 if close and peak_price else None
    return {
        "concentration": concentration,
        "winner": winner,
        "peak_price": peak_price,
        "peak_pctile": peak_pctile,
        "price_to_peak": price_to_peak,
    }


def _score_chip(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """信号③低位筹码集中：单峰密集 + 主峰低位 + 价格贴近主峰 + 获利盘中低位。"""
    m = _chip_metrics(bars)
    if m is None:
        return None, None
    conc = _clip01((m["concentration"] - CHIP_CONC_LO) / (CHIP_CONC_HI - CHIP_CONC_LO)) * 100.0
    peak_low = _clip01((CHIP_PEAK_ZERO - m["peak_pctile"]) / (CHIP_PEAK_ZERO - CHIP_PEAK_FULL)) * 100.0
    dist = m.get("price_to_peak")
    if dist is None or dist <= CHIP_PRICE_BELOW_ZERO or dist >= CHIP_PRICE_ABOVE_ZERO:
        price_near_peak = 0.0
    elif dist <= 0:
        price_near_peak = _clip01((dist - CHIP_PRICE_BELOW_ZERO) / (0.0 - CHIP_PRICE_BELOW_ZERO)) * 100.0
    elif dist <= CHIP_PRICE_ABOVE_FULL:
        price_near_peak = 100.0
    else:
        price_near_peak = _clip01((CHIP_PRICE_ABOVE_ZERO - dist) / (CHIP_PRICE_ABOVE_ZERO - CHIP_PRICE_ABOVE_FULL)) * 100.0

    winner = m["winner"]
    if winner <= CHIP_WINNER_MIN or winner >= CHIP_WINNER_MAX:
        winner_mid_low = 0.0
    elif winner < CHIP_WINNER_FULL_LO:
        winner_mid_low = _clip01((winner - CHIP_WINNER_MIN) / (CHIP_WINNER_FULL_LO - CHIP_WINNER_MIN)) * 100.0
    elif winner <= CHIP_WINNER_FULL_HI:
        winner_mid_low = 100.0
    else:
        winner_mid_low = _clip01((CHIP_WINNER_MAX - winner) / (CHIP_WINNER_MAX - CHIP_WINNER_FULL_HI)) * 100.0

    score = 0.45 * conc + 0.25 * peak_low + 0.20 * price_near_peak + 0.10 * winner_mid_low
    m["sub_concentration"] = conc
    m["sub_peak_low"] = peak_low
    m["sub_price_near_peak"] = price_near_peak
    m["sub_winner_mid_low"] = winner_mid_low
    return score, m


def _sealed_and_streak(bars: List[Dict[str, Any]], code: str) -> Tuple[int, int]:
    """近 SHORT_WIN 日一字封死板数量；以及当前连续涨停 streak 长度。"""
    limit = _limit_pct(code) - 0.3
    sealed = 0
    for i in range(max(1, len(bars) - SHORT_WIN), len(bars)):
        chg, prev_close = bars[i]["chg"], bars[i - 1]["close"]
        high, low = bars[i]["high"], bars[i]["low"]
        if chg is None or not prev_close or high is None or low is None:
            continue
        if chg >= limit and (high - low) / prev_close <= SEALED_AMP:
            sealed += 1
    streak = 0
    for b in reversed(bars):
        if b["chg"] is not None and b["chg"] >= limit:
            streak += 1
        else:
            break
    return sealed, streak


def _breakout_trigger(bars: List[Dict[str, Any]], vol: List[Optional[float]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """右进左出触发器：从低/中位整理中放量突破近 TRIGGER_WIN 日收盘高点。

    左侧吸筹分给「在不在吸筹」的状态；触发器给「是否刚启动」的买点——只有当价格突破
    一个『位于近 60 日中低位的整理平台』且当日放量，才算右侧确认(避免高位追涨/缩量假突破)。
    """
    if len(bars) < HIGH_WIN + 1:
        return False, None
    closes = [b["close"] for b in bars]
    c = closes[-1]
    prior = [x for x in closes[-1 - TRIGGER_WIN:-1] if x]
    if not c or len(prior) < TRIGGER_WIN // 2:
        return False, None
    ceiling = max(prior)
    broke = c > ceiling
    base_vol = _mean([v for v in vol[-1 - TRIGGER_WIN:-1] if v])
    vol_confirm = bool(base_vol and vol[-1] and vol[-1] > TRIGGER_VOL_MULT * base_vol)
    win = [x for x in closes[-HIGH_WIN - 1:-1] if x]   # 突破前的近 60 日(不含今日)
    base_pctile = sum(1 for x in win if x <= ceiling) / len(win) if win else 1.0
    from_low = base_pctile < TRIGGER_BASE_MAX_POS
    triggered = bool(broke and vol_confirm and from_low)
    return triggered, {"breakout": bool(broke), "vol_confirm": vol_confirm, "base_pctile": round(base_pctile, 2)}


# ── 游资形态匹配器（P1-P26）──────────────────────────────────────
# 每个匹配器输入 (bars, ctx) 返回 bool（命中），PIT 安全（只用窗口内 bar）。
# 信号方向：buy=吸筹/洗盘(左侧) · hold=拉升(只标记不追) · sell=出货(风控/回避)
#          · observe=中性结构观察（不驱动主导阶段）。

def _kl_pc(bars: List[Dict[str, Any]], i: int) -> Optional[float]:
    return bars[i - 1]["close"] if i > 0 else None


def _kl_amp(bars, i):
    pc, h, l = _kl_pc(bars, i), bars[i]["high"], bars[i]["low"]
    return (h - l) / pc if pc and h is not None and l is not None else None


def _kl_body(bars, i):
    pc, o, c = _kl_pc(bars, i), bars[i]["open"], bars[i]["close"]
    return abs(c - o) / pc if pc and o is not None and c is not None else None


def _kl_upper(bars, i):
    pc, h, o, c = _kl_pc(bars, i), bars[i]["high"], bars[i]["open"], bars[i]["close"]
    return (h - max(o, c)) / pc if pc and None not in (h, o, c) else None


def _kl_lower(bars, i):
    pc, l, o, c = _kl_pc(bars, i), bars[i]["low"], bars[i]["open"], bars[i]["close"]
    return (min(o, c) - l) / pc if pc and None not in (l, o, c) else None


def _kl_doji(bars, i):
    b = _kl_body(bars, i)
    return b is not None and b < 0.005


def _ret_k(closes: List[Optional[float]], k: int) -> Optional[float]:
    if len(closes) > k and closes[-1] and closes[-1 - k]:
        return closes[-1] / closes[-1 - k] - 1
    return None


def _ma_last(closes: List[Optional[float]], n: int) -> Optional[float]:
    vals = [c for c in closes[-n:] if c is not None]
    return sum(vals) / len(vals) if len(vals) >= max(2, int(n * 0.6)) else None


def _avg_vol(vol: List[Optional[float]], a: int, b: int) -> Optional[float]:
    seg = [v for v in vol[a:b] if v]
    return sum(seg) / len(seg) if seg else None


def _swing_lows(values: List[float], k: int = 3) -> List[int]:
    """局部低点(摆动低)索引：values[i] 是前后各 k 根内的最小值。

    用于底部形态识别（双底/W底的两个谷）；相邻 k 根内只保留更低的一个，避免平台重复计点。
    """
    lows: List[int] = []
    n = len(values)
    for i in range(k, n - k):
        v = values[i]
        if v is None or v != min(values[i - k:i + k + 1]):
            continue
        if lows and i - lows[-1] <= k:        # 太近：保留更低者
            if v < values[lows[-1]]:
                lows[-1] = i
        else:
            lows.append(i)
    return lows


_PRIOR_CHIP_UNSET = object()


def _build_pattern_context(
    code: str,
    bars: List[Dict[str, Any]],
    prior_chip: Any = _PRIOR_CHIP_UNSET,
    defer_chip: bool = False,
) -> Dict[str, Any]:
    """一次性算齐形态匹配要用的量价上下文（避免每个匹配器各算一遍）。"""
    current_bars = bars[-LOOKBACK:]
    closes = [b["close"] for b in current_bars]
    vol, vol_measure = _volume_series(current_bars)
    score_position, pos = _score_position(current_bars)
    _, vol_ratio = _score_volume_ratio(vol)
    day_volume_ratio = _p26_volume_ratio(vol)
    score_divergence, drift = _score_divergence(current_bars, vol)
    score_cmf, cmf = _score_cmf(current_bars, vol)
    if prior_chip is _PRIOR_CHIP_UNSET:
        prior_end = len(bars) - CHIP_WINNER_RISK_DAYS
        prior_start = max(0, prior_end - LOOKBACK)
        prior_chip_bars = bars[prior_start:prior_end]
    else:
        prior_chip_bars = []
    if defer_chip:
        score_chip, chip = None, None
        prior_chip_value = None if prior_chip is _PRIOR_CHIP_UNSET else prior_chip
    else:
        score_chip, chip = _score_chip(current_bars)
        if prior_chip is _PRIOR_CHIP_UNSET:
            prior_chip_value = (
                _chip_metrics(prior_chip_bars)
                if len(prior_chip_bars) >= MIN_BARS
                else None
            )
        else:
            prior_chip_value = prior_chip
    chip_winner = chip.get("winner") if chip else None
    chip_winner_prior = prior_chip_value.get("winner") if prior_chip_value else None
    chip_winner_change = (
        chip_winner - chip_winner_prior
        if chip_winner is not None and chip_winner_prior is not None
        else None
    )
    sealed, streak = _sealed_and_streak(current_bars, code)
    triggered, trigger = _breakout_trigger(current_bars, vol)
    ma = {n: _ma_last(closes, n) for n in (5, 10, 20, 60)}
    ma_bull = bool(ma[5] and ma[10] and ma[20] and ma[5] > ma[10] > ma[20]
                   and closes[-1] and closes[-1] > ma[5])
    context = {
        "code": code, "closes": closes, "vol": vol,
        "pos": pos, "vol_ratio": vol_ratio, "drift": drift, "cmf": cmf,
        "chip": chip, "sealed": sealed, "streak": streak, "triggered": triggered,
        "vol_measure": vol_measure,
        "score_position": score_position,
        "score_divergence": score_divergence,
        "score_cmf": score_cmf,
        "score_chip": score_chip,
        "trigger": trigger,
        "chip_winner": chip_winner,
        "chip_winner_prior": chip_winner_prior,
        "chip_winner_change": chip_winner_change,
        "day_volume_ratio": day_volume_ratio,
        "p26_volume_ratio": day_volume_ratio,
        "ma5": ma[5], "ma10": ma[10], "ma20": ma[20], "ma60": ma[60], "ma_bull": ma_bull,
    }
    if defer_chip:
        context.update({
            "_chip_deferred": True,
            "_chip_computed": False,
            "_prior_chip_computed": prior_chip is not _PRIOR_CHIP_UNSET,
            "_prior_chip": prior_chip_value,
            "_chip_bars": current_bars,
            "_prior_chip_bars": prior_chip_bars,
        })
    return context


def _ensure_pattern_chip_context(
    ctx: Dict[str, Any],
    include_prior: bool = False,
) -> Optional[Dict[str, float]]:
    """按需补算形态研究的筹码上下文；实时评分仍走 eager 路径。"""
    if not ctx.get("_chip_deferred"):
        return ctx.get("chip")
    if not ctx.get("_chip_computed"):
        score_chip, chip = _score_chip(ctx.get("_chip_bars") or [])
        ctx["score_chip"] = score_chip
        ctx["chip"] = chip
        ctx["chip_winner"] = chip.get("winner") if chip else None
        ctx["_chip_computed"] = True
    if include_prior and not ctx.get("_prior_chip_computed"):
        prior_bars = ctx.get("_prior_chip_bars") or []
        ctx["_prior_chip"] = _chip_metrics(prior_bars) if len(prior_bars) >= MIN_BARS else None
        ctx["_prior_chip_computed"] = True
    if include_prior:
        prior_chip = ctx.get("_prior_chip")
        ctx["chip_winner_prior"] = prior_chip.get("winner") if prior_chip else None
        winner = ctx.get("chip_winner")
        prior = ctx.get("chip_winner_prior")
        ctx["chip_winner_change"] = (
            winner - prior if winner is not None and prior is not None else None
        )
    return ctx.get("chip")


# --- 吸筹 🟢buy ---
def _p1_low_base_state(bars: List[Dict[str, Any]]) -> bool:
    """P1原始状态：低位窄幅横住、贴近MA20、收阳且温和缩量。"""
    if len(bars) < HIGH_WIN:
        return False
    current_bars = bars[-LOOKBACK:]
    try:
        closes = [float(bar.get("close")) for bar in current_bars]
        open_price = float(current_bars[-1].get("open"))
    except (TypeError, ValueError):
        return False
    if (
        len(closes) < HIGH_WIN
        or any(not math.isfinite(value) or value <= 0 for value in closes[-HIGH_WIN:])
        or not math.isfinite(open_price)
        or open_price <= 0
    ):
        return False

    close = closes[-1]
    position60 = sum(1 for value in closes[-HIGH_WIN:] if value <= close) / HIGH_WIN
    if position60 >= P1_POSITION_60_MAX:
        return False

    ma20 = _mean(closes[-20:])
    reference20 = closes[-21] if len(closes) > 20 else None
    if (
        ma20 is None
        or not reference20
        or abs(close / reference20 - 1.0) > P1_ABS_RET_20_MAX
        or close < P1_MA20_FLOOR * ma20
        or close < open_price
    ):
        return False

    volume, _source = _volume_series(current_bars)
    day_volume_ratio = _p26_volume_ratio(volume)
    return bool(
        day_volume_ratio is not None
        and P1_DAY_VOLUME_RATIO_MIN <= day_volume_ratio <= P1_DAY_VOLUME_RATIO_MAX
    )


def _p1_breakout_addon_state(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算P1突破补充腿的个股侧状态，先做纯突破去重，再叠加过滤条件。

    “昨日是否已突破”只看昨日相对其此前10日收盘的纯突破状态。即使昨日因量能、
    位置或市场门未通过，今天仍不会被误当作一个新的突破 episode。
    """
    empty: Dict[str, Any] = {
        "available": False,
        "current_breakout": False,
        "previous_breakout": False,
        "first_breakout": False,
        "stock_gate": False,
        "position60": None,
        "ret20": None,
        "close_over_ma20": None,
        "day_volume_ratio": None,
        "day_volume_source": None,
        "reason": "insufficient_history",
        "failed_checks": ["history"],
    }
    required = max(HIGH_WIN, P1_BREAKOUT_DAYS + 2, 21)
    current_bars = bars[-LOOKBACK:]
    if len(current_bars) < required:
        return empty

    closes = [_safe(bar.get("close")) for bar in current_bars]
    recent = closes[-HIGH_WIN:]
    if any(value is None or value <= 0 for value in recent):
        invalid = dict(empty)
        invalid["reason"] = "invalid_close"
        invalid["failed_checks"] = ["valid_close"]
        return invalid

    # All closes needed below are inside the validated trailing 60-bar window.
    close = float(closes[-1])
    previous_close = float(closes[-2])
    prior_today = [float(value) for value in closes[-(P1_BREAKOUT_DAYS + 1):-1]]
    prior_yesterday = [float(value) for value in closes[-(P1_BREAKOUT_DAYS + 2):-2]]
    current_breakout = close > max(prior_today)
    previous_breakout = previous_close > max(prior_yesterday)
    first_breakout = bool(current_breakout and not previous_breakout)

    position60 = sum(1 for value in recent if float(value) <= close) / HIGH_WIN
    reference20 = float(closes[-21])
    ret20 = close / reference20 - 1.0
    ma20 = _mean([float(value) for value in closes[-20:]])
    close_over_ma20 = close / ma20 if ma20 and ma20 > 0 else None
    volume, volume_source = _volume_series(current_bars)
    day_volume_ratio = _p26_volume_ratio(volume)

    checks = {
        "first_breakout": first_breakout,
        "position60": position60 < P1_ADDON_POSITION_60_MAX,
        "ret20": abs(ret20) <= P1_ADDON_ABS_RET_20_MAX,
        "ma20": bool(
            close_over_ma20 is not None
            and close_over_ma20 >= P1_ADDON_MA20_FLOOR
        ),
        "volume": bool(
            day_volume_ratio is not None
            and P1_ADDON_DAY_VOLUME_RATIO_MIN
            <= day_volume_ratio
            <= P1_ADDON_DAY_VOLUME_RATIO_MAX
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "available": True,
        "current_breakout": bool(current_breakout),
        "previous_breakout": bool(previous_breakout),
        "first_breakout": first_breakout,
        "stock_gate": not failed,
        "position60": position60,
        "ret20": ret20,
        "close_over_ma20": close_over_ma20,
        "day_volume_ratio": day_volume_ratio,
        "day_volume_source": volume_source,
        "reason": "open" if not failed else "closed",
        "failed_checks": failed,
    }


def _p1_rule_state(
    bars: List[Dict[str, Any]],
    market_gate: bool,
) -> Dict[str, Any]:
    """正式P1复合事件：原两日企稳腿 OR 首次10日突破补充腿。"""
    base_hit = _confirmed_new_state(
        bars, _p1_low_base_state, days=P1_CONFIRM_DAYS
    )
    stock = _p1_breakout_addon_state(bars)
    supplemental_hit = bool(stock["stock_gate"] and market_gate)
    active = bool(base_hit or supplemental_hit)
    if base_hit and supplemental_hit:
        source = "base_and_breakout_addon"
    elif base_hit:
        source = "base_leg"
    elif supplemental_hit:
        source = "breakout_addon"
    else:
        source = None
    signal_date = str(bars[-1].get("date") or "") if bars else None
    return {
        "code": "P1",
        "name": "P1超额优先复合确认",
        "version": P1_VERSION,
        "replaced_on": P1_REPLACED_ON,
        "signal_date": signal_date,
        "active": active,
        "hit": active,
        "source": source,
        "base_leg_hit": bool(base_hit),
        "supplemental_hit": supplemental_hit,
        "market_gate": bool(market_gate),
        "breakout_addon_state": stock,
        "eod_only": True,
        "score_weight": ACCUM_MODEL_WEIGHTS["p1"],
        "production_replacement": True,
    }


def _pat_low_consolidation(
    bars: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    p1_market_gate: bool = False,
) -> bool:                                  # P1 超额优先复合确认
    return bool(_p1_rule_state(bars, p1_market_gate)["active"])


def _p1_rule_params() -> Dict[str, Any]:
    """正式P1生产/回测共用的可审计参数快照。"""
    return {
        "code": "P1",
        "name": "P1超额优先复合确认",
        "version": P1_VERSION,
        "replaced_on": P1_REPLACED_ON,
        "scope": "leader_and_hotmoney",
        "validated_scope": "leader_and_hotmoney",
        "primary_horizons": list(VERIFY_HORIZONS),
        "etf_policy": "base_leg_only; breakout addon is disabled pending ETF-specific validation",
        "production_replacement": True,
        "eod_only": True,
        "score_weight": ACCUM_MODEL_WEIGHTS["p1"],
        "holding_days": 10,
        "union_rule": "base leg OR supplemental breakout leg",
        "base_leg": {
            "position60_max_exclusive": P1_POSITION_60_MAX,
            "abs_ret20_max": P1_ABS_RET_20_MAX,
            "close_over_ma20_min": P1_MA20_FLOOR,
            "close_at_or_above_open": True,
            "day_volume_ratio": [P1_DAY_VOLUME_RATIO_MIN, P1_DAY_VOLUME_RATIO_MAX],
            "confirmation_days": P1_CONFIRM_DAYS,
            "event_rule": "first day on which the raw state has held for two consecutive bars",
        },
        "supplemental_leg": {
            "breakout_close_days": P1_BREAKOUT_DAYS,
            "event_rule": (
                "today close is strictly above the prior 10-close high and "
                "yesterday was not itself above its prior 10-close high"
            ),
            "position60_max_exclusive": P1_ADDON_POSITION_60_MAX,
            "abs_ret20_max": P1_ADDON_ABS_RET_20_MAX,
            "close_over_ma20_min": P1_ADDON_MA20_FLOOR,
            "day_volume_ratio": [
                P1_ADDON_DAY_VOLUME_RATIO_MIN,
                P1_ADDON_DAY_VOLUME_RATIO_MAX,
            ],
            "green_candle_required": False,
        },
        "day_volume_ratio_base_days": CHIP_WINNER_RISK_VOLUME_BASE_WIN,
        "day_volume_source": "turnover if PIT window coverage >=70%, otherwise volume",
        "market_gate": {
            "applies_to": "supplemental_leg_only",
            "index": MARKET_REGIME_INDEX,
            "value_source": "nav_acc preferred, nav fallback",
            "same_date_required": True,
            "minimum_history": 60,
            "close_over_ma20_min": P1_MARKET_MA20_FLOOR,
            "close_over_ma60_min": P1_MARKET_MA60_FLOOR,
            "ret5": [P1_MARKET_RET5_MIN, P1_MARKET_RET5_MAX],
            "ret20": [P1_MARKET_RET20_MIN, P1_MARKET_RET20_MAX],
            "ma20_slope5_min": P1_MARKET_MA20_SLOPE5_MIN,
            "failure_policy": "fail_closed",
        },
        "chip_gate": None,
        "backtest_snapshot": {
            "leader": {
                "events": 9528,
                "coverage": 0.005185,
                "win_rate_10d": 0.5479,
                "excess_10d": 0.00464,
                "hac_t": 3.21,
            },
            "hotmoney": {
                "events": 2552,
                "coverage": 0.005085,
                "win_rate_10d": 0.5756,
                "excess_10d": 0.01297,
                "hac_t": 3.24,
            },
        },
    }


def _pat_low_shadows(bars, ctx):                 # P2 低位真实长下影+温和缩量承接
    if ctx["pos"] is None or ctx["pos"] >= 0.40 or len(bars) < P2_WINDOW + 1:
        return False
    day_volume_ratio = ctx.get("p26_volume_ratio")
    if day_volume_ratio is None:
        volume = ctx.get("vol")
        if volume is None:
            volume, _source = _volume_series(bars)
        day_volume_ratio = _p26_volume_ratio(volume)
    if (
        day_volume_ratio is None
        or day_volume_ratio < P2_DAY_VOLUME_RATIO_MIN
        or day_volume_ratio > P2_DAY_VOLUME_RATIO_MAX
    ):
        return False
    matched: List[bool] = []
    for i in range(len(bars) - P2_WINDOW, len(bars)):
        body, low = _kl_body(bars, i), _kl_lower(bars, i)
        matched.append(bool(
            body is not None
            and low is not None
            and low >= P2_SHADOW_BODY_RATIO_MIN * body
            and low >= P2_LOWER_SHADOW_MIN
        ))
    return matched[-1] and sum(matched) >= P2_MIN_COUNT


def _pat_shakedown_absorb(bars, ctx):            # P3 隐性收集(缩量收盘大跌后首次收复)
    pos, vol, closes = ctx["pos"], ctx["vol"], ctx["closes"]
    if pos is None or pos >= P3_POSITION_MAX or len(bars) < 30:
        return False
    confirm_volume_ratio = ctx.get("p26_volume_ratio")
    if confirm_volume_ratio is None:
        confirm_volume_ratio = _p26_volume_ratio(vol)
    if confirm_volume_ratio is None or confirm_volume_ratio > P3_CONFIRM_VOLUME_RATIO_MAX:
        return False
    for i in range(len(bars) - 2, len(bars) - P3_EVENT_WINDOW - 2, -1):
        pre_drop, drop_close = closes[i - 1], closes[i]
        if (
            not pre_drop or not drop_close
            or drop_close > pre_drop * (1.0 - P3_DROP_MIN)
        ):
            continue
        # 绝大多数交易日并非大跌事件：先做O(1)价格门，再计算该事件日前20日均量。
        event_volume = vol[i]
        event_base = _avg_vol(vol, i - 20, i)
        if (
            not event_volume or not event_base
            or event_volume >= P3_EVENT_VOLUME_RATIO_MAX * event_base
        ):
            continue
        # 真正首次收复：事件后到昨日从未站回跌前收盘；同时保留宽松的6%破底保护。
        if not closes[-1] or not closes[-2] or closes[-1] < pre_drop or closes[-2] >= pre_drop:
            continue
        if any(
            not value
            or value >= pre_drop
            or value < drop_close * (1.0 - P3_UNDERCUT_MAX)
            for value in closes[i + 1:-1]
        ):
            continue
        return True
    return False


def _p4_absorption_state(bars: List[Dict[str, Any]]) -> Optional[bool]:
    """P4核心状态；固定180日窗口，数据不足返回None以便首次事件严格失效。"""
    if len(bars) < LOOKBACK:
        return None
    current_bars = bars[-LOOKBACK:]
    _, pos = _score_position(current_bars)
    volume, _source = _volume_series(current_bars)
    _, volume_ratio = _score_volume_ratio(volume)
    _, drift = _score_absorption(current_bars)
    _, cmf = _score_cmf(current_bars, volume)
    if any(value is None for value in (pos, volume_ratio, drift, cmf)):
        return None
    return bool(
        pos < P4_POSITION_MAX
        and P4_VOLUME_RATIO_MIN < volume_ratio <= P4_VOLUME_RATIO_MAX
        and abs(drift) < P4_ABS_DRIFT_MAX
        and cmf > P4_CMF_MIN
    )


def _p4_current_state(ctx: Dict[str, Any]) -> bool:
    pos, vr, drift, cmf = ctx["pos"], ctx["vol_ratio"], ctx["drift"], ctx["cmf"]
    return bool(
        pos is not None
        and pos < P4_POSITION_MAX
        and vr is not None
        and P4_VOLUME_RATIO_MIN < vr <= P4_VOLUME_RATIO_MAX
        and drift is not None
        and abs(drift) < P4_ABS_DRIFT_MAX
        and cmf is not None
        and cmf > P4_CMF_MIN
    )


def _pat_absorption(bars, ctx):                  # P4 量增价稳(首次观察事件)
    """只报核心状态由昨日不成立到今日成立，避免同一放量平台逐日重复命中。"""
    if len(bars) < LOOKBACK + 1 or not _p4_current_state(ctx):
        return False
    previous_state = _p4_absorption_state(bars[:-1])
    return previous_state is False


def _p4_rule_params() -> Dict[str, Any]:
    """P4生产/回测共用的可审计参数快照。"""
    return {
        "scope": "leader_and_hotmoney",
        "validation_status": "observation_only",
        "position60_max_exclusive": P4_POSITION_MAX,
        "volume_ratio": {
            "min_exclusive": P4_VOLUME_RATIO_MIN,
            "max_inclusive": P4_VOLUME_RATIO_MAX,
            "recent_days": SHORT_WIN,
            "base_days": BASE_WIN,
        },
        "abs_ret20_max_exclusive": P4_ABS_DRIFT_MAX,
        "cmf20_min_exclusive": P4_CMF_MIN,
        "event_rule": "first day on which the core state changes from false to true",
        "unknown_previous_state": "fail_closed",
        "additional_requirement_count": 1,
        "signal_semantics": "neutral observation; does not drive bullish phase",
    }


def _pat_bottom_formation(bars, ctx):            # P5 双底右腿确认
    """低位W底的右腿确认事件，而不是二底后每天重复成立的状态。

    两个新增要求均为短窗口价格比较：二底须在近8日，且今日由“昨日未突破”切换为突破
    前3日收盘高点。原有“二底回升”下界收紧为触及颈线；不再重建筹码分布。
    """
    pos, closes = ctx["pos"], ctx["closes"]
    if pos is None or pos >= P5_POSITION_MAX:
        return False
    cs = []
    for value in closes[-60:]:
        try:
            close = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(close) or close <= 0:
            return False
        cs.append(close)
    if len(cs) < 40:
        return False
    lows = _swing_lows(cs, k=3)
    if len(lows) < 2:
        return False
    i1, i2 = lows[-2], lows[-1]
    if i2 - i1 < 5:                               # 两底间隔太近不算结构
        return False
    l1, l2 = cs[i1], cs[i2]
    second_low_return = l2 / l1 - 1.0
    if (second_low_return < P5_SECOND_LOW_RETURN_MIN - P5_FLOAT_EPSILON
            or second_low_return > P5_SECOND_LOW_RETURN_MAX + P5_FLOAT_EPSILON):
        return False
    neck = max(cs[i1:i2 + 1])                     # 两底之间的反弹高点=颈线
    if neck / min(l1, l2) - 1 < P5_NECK_REBOUND_MIN - P5_FLOAT_EPSILON:
        return False
    if len(cs) - 1 - i2 > P5_SECOND_LOW_RECENCY_MAX:
        return False                               # 只看二底确认后的近端右腿
    c = cs[-1]
    neck_ratio = c / neck
    if (neck_ratio < P5_NECK_RATIO_MIN - P5_FLOAT_EPSILON
            or neck_ratio > P5_NECK_RATIO_MAX + P5_FLOAT_EPSILON):
        return False
    win = P5_RIGHT_LEG_BREAKOUT_WIN
    current_breakout = c > max(cs[-win - 1:-1])
    previous_breakout = cs[-2] > max(cs[-win - 2:-2])
    return current_breakout and not previous_breakout


def _p5_rule_params() -> Dict[str, Any]:
    """P5生产/回测共用的可审计参数快照。"""
    return {
        "scope": "leader_and_hotmoney",
        "validation_status": "10d_observation",
        "holding_days": 10,
        "position60_max_exclusive": P5_POSITION_MAX,
        "swing_low_k": 3,
        "second_low_return": [P5_SECOND_LOW_RETURN_MIN, P5_SECOND_LOW_RETURN_MAX],
        "neck_rebound_min": P5_NECK_REBOUND_MIN,
        "second_low_recency_max": P5_SECOND_LOW_RECENCY_MAX,
        "neck_ratio": [P5_NECK_RATIO_MIN, P5_NECK_RATIO_MAX],
        "right_leg_breakout_window": P5_RIGHT_LEG_BREAKOUT_WIN,
        "event_rule": "today closes above the prior 3-close high after yesterday did not",
        "additional_requirement_count": 2,
        "additional_requirements": [
            "second swing low occurred within the latest 8 trading days",
            "3-close-high breakout changes from false yesterday to true today",
        ],
        "chip_gate": None,
    }


def _confirmed_new_state(bars, predicate, days=PATTERN_CONFIRM_DAYS):
    """条件连续成立 ``days`` 天时只触发一次，避免状态型形态每天重复命中。"""
    if len(bars) <= days:
        return False
    # 绝大多数截面当前条件就不成立；短路可避免为每只股票无条件重算6遍滚动状态。
    if not predicate(bars):
        return False
    for offset in range(1, days):
        if not predicate(bars[:-offset]):
            return False
    return not predicate(bars[:-days])


def _p23_compression_state(bars: List[Dict[str, Any]]) -> Optional[bool]:
    """P23核心状态；数据不足返回None，避免把未知误当成“昨日未压缩”。"""
    if len(bars) < HIGH_WIN + 1:                 # 60个振幅至少需要61根bar
        return None
    _, pos = _score_position(bars)
    ratio = _amp_ratio(bars)
    if pos is None or ratio is None:
        return None
    return pos < P23_POSITION_60_MAX and ratio < COMPRESS_AMP_RATIO


def _p23_confirmation_state(
    bars: List[Dict[str, Any]],
) -> Optional[bool]:
    """P23结构确认状态：低位压缩且略站上MA20；数据不足返回None。"""
    core_state = _p23_compression_state(bars)
    if core_state is not True:
        return core_state

    current_bars = bars[-LOOKBACK:]
    closes = [_safe(bar.get("close")) for bar in current_bars[-20:]]
    if len(closes) < 20 or any(value is None or value <= 0 for value in closes):
        return None
    numeric_closes = [float(value) for value in closes if value is not None]
    ma20 = _mean(numeric_closes)
    close_over_ma20 = numeric_closes[-1] / ma20 if ma20 and ma20 > 0 else None
    return bool(close_over_ma20 is not None and close_over_ma20 >= P23_MA20_FLOOR)


def _p23_temperate_volume_state(
    bars: List[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """P23信号日量能门；不参与结构状态去重，避免量比进出区间造成重复触发。"""
    if ctx is not None and "p26_volume_ratio" in ctx:
        day_volume_ratio = ctx.get("p26_volume_ratio")
    else:
        current_bars = bars[-LOOKBACK:]
        volume, _source = _volume_series(current_bars)
        day_volume_ratio = _p26_volume_ratio(volume)
    if day_volume_ratio is None:
        return None
    return bool(
        P23_DAY_VOLUME_RATIO_MIN <= day_volume_ratio <= P23_DAY_VOLUME_RATIO_MAX
    )


def _pat_compression(bars, ctx):                 # P23 低位压缩温和量能转强
    """结构确认状态由False切为True后再检查当日量能；同一连续状态只报告一次。"""
    if len(bars) < HIGH_WIN + 2:                 # 昨日状态也必须有完整60个振幅
        return False
    current_state = _p23_confirmation_state(bars)
    if current_state is not True:
        return False
    if _p23_confirmation_state(bars[:-1]) is not False:
        return False
    return _p23_temperate_volume_state(bars, ctx=ctx) is True


def _p23_rule_params() -> Dict[str, Any]:
    """P23生产/回测共用的可审计参数快照。"""
    return {
        "scope": "leader_and_hotmoney",
        "validated_scope": "experimental_dual_pool",
        "holding_days": 10,
        "position60_max_exclusive": P23_POSITION_60_MAX,
        "amp20_over_amp60_max_exclusive": COMPRESS_AMP_RATIO,
        "close_over_ma20_min": P23_MA20_FLOOR,
        "day_volume_ratio": [P23_DAY_VOLUME_RATIO_MIN, P23_DAY_VOLUME_RATIO_MAX],
        "day_volume_ratio_base_days": CHIP_WINNER_RISK_VOLUME_BASE_WIN,
        "day_volume_source": "turnover if trailing-180 PIT coverage >=70%, otherwise volume",
        "event_rule": "first day entering the compression + MA20 structural state",
        "volume_gate_scope": "signal day only; excluded from event-state deduplication",
        "max_extra_requirements": 2,
        "incremental_requirement_count_vs_previous_p23": 1,
        "incremental_requirements": [
            "signal-day volume over prior-20-day mean must be within [0.40, 1.20]",
        ],
    }


def _p24_obv_divergence_state(bars):
    """P24 原始状态：中低位、价格未启动但30日OBV有显著净流入。"""
    if len(bars) < HIGH_WIN:
        return False
    _, pos = _score_position(bars)
    closes = [bar.get("close") for bar in bars]
    volumes, _source = _volume_series(bars)
    if pos is None or pos >= OBV_POSITION_MAX or len(closes) < 31:
        return False
    if not closes[-1] or not closes[-31]:
        return False
    obv = 0.0
    for i in range(len(closes) - 30, len(closes)):
        if closes[i] is None or closes[i - 1] is None:
            continue
        direction = 1 if closes[i] > closes[i - 1] else (-1 if closes[i] < closes[i - 1] else 0)
        obv += direction * (volumes[i] or 0)
    valid_volume = [value for value in volumes[-60:] if value]
    mean_volume = (sum(valid_volume) / len(valid_volume)) if valid_volume else 0.0
    if not mean_volume:
        return False
    price_return = closes[-1] / closes[-31] - 1.0
    obv_return = obv / (mean_volume * 30)
    return (
        price_return <= OBV_PRICE_RETURN_MAX
        and obv_return > OBV_RETURN_MIN
        and obv_return - price_return > OBV_DIV_MIN
    )


def _pat_obv_divergence(bars, ctx):              # P24 OBV底背离(价弱量增)
    """严格OBV底背离连续5日确认后仅触发一次，过滤短暂噪声与重复命中。"""
    return _confirmed_new_state(bars, _p24_obv_divergence_state)


def _pat_bottom_base_ignition(bars, ctx):         # P25 低位缩量平台转强
    """120日中低位 + 60日横盘缩量，随后价格小幅转强并接近20日平台。

    用价格确认替代旧版爆量确认；所有股票池口径相同，计算量不增加。
    """
    if len(bars) < 120:
        return False
    closes = [b.get("close") for b in bars]
    if any(value is None or value <= 0 for value in closes[-121:]):
        return False
    ret5 = closes[-1] / closes[-6] - 1.0
    recent_chg = [bars[i].get("chg") for i in range(len(bars) - 3, len(bars))]
    if any(value is None for value in recent_chg):
        return False
    if ret5 > 0.08 or bars[-1]["chg"] >= 6.0 or max(recent_chg) >= 6.0:
        return False

    close = closes[-1]
    position120 = sum(1 for value in closes[-120:] if value <= close) / 120.0
    if position120 > P25_POSITION_120_MAX:
        return False
    highs = [b.get("high") for b in bars[-60:]]
    lows = [b.get("low") for b in bars[-60:]]
    if any(value is None or value <= 0 for value in highs + lows):
        return False
    range60 = max(highs) / min(lows) - 1.0
    abs_ret60 = abs(close / closes[-61] - 1.0)
    if range60 > P25_RANGE_60_MAX or abs_ret60 > P25_ABS_RET_60_MAX:
        return False
    ret20 = close / closes[-21] - 1.0
    if ret20 < P25_RET_20_MIN:
        return False

    volumes = [b.get("volume") for b in bars]
    if any(value is None or value <= 0 for value in volumes[-64:]):
        return False
    early40 = _mean(volumes[-60:-20])
    recent20 = _mean(volumes[-20:])
    if not early40 or not recent20:
        return False
    if recent20 / early40 > P25_VOLUME_CONTRACT_MAX:
        return False

    volume_ma5 = [_mean(volumes[i - 4:i + 1]) for i in range(len(volumes) - 60, len(volumes))]
    axis = list(range(60))
    axis_mean = 29.5
    denom = sum((value - axis_mean) ** 2 for value in axis)
    volume_mean = _mean(volume_ma5)
    slope = sum((x - axis_mean) * y for x, y in zip(axis, volume_ma5)) / denom
    normalized_slope = slope / volume_mean if volume_mean else 0.0
    if normalized_slope > P25_VOLUME_SLOPE_60_MAX:
        return False

    prior20 = [value for value in closes[-21:-1] if value]
    breakout20 = close / max(prior20) - 1.0 if prior20 else -1.0
    return breakout20 >= P25_BREAKOUT_20_MIN


def _p25_rule_params() -> Dict[str, Any]:
    """P25生产/回测共用的可审计参数快照。"""
    return {
        "scope": "leader_and_hotmoney",
        "validation_status": "experimental_dual_pool_directional",
        "position120_max": P25_POSITION_120_MAX,
        "range60_max": P25_RANGE_60_MAX,
        "abs_ret60_max": P25_ABS_RET_60_MAX,
        "volume_contract_20_60_max": P25_VOLUME_CONTRACT_MAX,
        "volume_slope60_max": P25_VOLUME_SLOPE_60_MAX,
        "ret20_min": P25_RET_20_MIN,
        "breakout20_min": P25_BREAKOUT_20_MIN,
        "anti_chase": {"ret5_max": 0.08, "max_change3_pct_exclusive": 6.0},
        "replaced_requirements": [
            "volume_last_vs_prev20 >= 1.5 -> ret20 >= 0.005",
            "breakout20 >= -0.02 -> breakout20 >= -0.04",
        ],
        "additional_requirement_count": 0,
        "primary_horizons": [10, 20],
    }


def _p26_peak_drawdown_confirmed(
    close: Optional[float],
    peak_high: Optional[float],
) -> bool:
    """P26峰值回撤边界；研究脚本使用相同的乘法表达式。"""
    return bool(
        close is not None
        and close > 0.0
        and peak_high is not None
        and peak_high > 0.0
        and close <= (1.0 - P26_PEAK_DRAWDOWN_MIN) * peak_high
    )


def _p26_price_reversal_confirmed(
    bars: Sequence[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]] = None,
) -> bool:
    """P26廉价价格门：先快速拉升，再从近端盘中高点明确回撤。"""
    if len(bars) < max(P26_PEAK_HIGH_DAYS, 6):
        return False
    closes = list((ctx or {}).get("closes") or [bar.get("close") for bar in bars])
    ret5 = _ret_k(closes, 5)
    close = _safe(bars[-1].get("close"))
    recent_highs = [
        value
        for value in (
            _safe(bar.get("high")) for bar in bars[-P26_PEAK_HIGH_DAYS:]
        )
        if value is not None and value > 0.0
    ]
    return bool(
        ret5 is not None
        and ret5 >= P26_RET_5_MIN
        and recent_highs
        and _p26_peak_drawdown_confirmed(close, max(recent_highs))
    )


def _pat_chip_winner_risk(bars, ctx):             # P26 获利盘冲高回撤风险
    """获利筹码拥挤/快速转盈、极端放量且冲高回撤，提示兑现或派发风险。"""
    volume_ratio = ctx.get("p26_volume_ratio")
    if volume_ratio is None or volume_ratio < CHIP_WINNER_RISK_VOLUME_RATIO_MIN:
        return False
    # 先检查O(1)/固定10根的价格门，再做昂贵的筹码重建。稳态上升和放量续涨
    # 不应仅因获利盘高而过早卖出；已有冲高、且价格离开峰值后才升级风险。
    if not _p26_price_reversal_confirmed(bars, ctx=ctx):
        return False
    _ensure_pattern_chip_context(ctx, include_prior=True)
    winner = ctx.get("chip_winner")
    prior = ctx.get("chip_winner_prior")
    return bool(
        (winner is not None and winner >= CHIP_WINNER_RISK_HIGH)
        or (
            winner is not None and prior is not None
            and prior < CHIP_WINNER_RISK_PRIOR_MAX
            and winner >= CHIP_WINNER_RISK_CURRENT_MIN
        )
    )


def _p26_rule_params() -> Dict[str, Any]:
    """P26冻结生产口径，供实时结果、回测和测试共同审计。"""
    return {
        "code": "P26",
        "name": "P26获利盘冲高回撤风险",
        "scope": "leader_and_hotmoney",
        "signal": "sell",
        "chip_union": {
            "winner_min_inclusive": CHIP_WINNER_RISK_HIGH,
            "jump_prior_max_exclusive": CHIP_WINNER_RISK_PRIOR_MAX,
            "jump_current_min_inclusive": CHIP_WINNER_RISK_CURRENT_MIN,
            "lookback_sessions": CHIP_WINNER_RISK_DAYS,
        },
        "day_volume_ratio_min_inclusive": CHIP_WINNER_RISK_VOLUME_RATIO_MIN,
        "day_volume_ratio_base_days": CHIP_WINNER_RISK_VOLUME_BASE_WIN,
        "day_volume_ratio_min_observations": CHIP_WINNER_RISK_VOLUME_MIN_OBS,
        "day_volume_source": (
            "turnover if trailing-180 PIT coverage >=70%, otherwise volume"
        ),
        "ret5_min_inclusive": P26_RET_5_MIN,
        "peak_high_days": P26_PEAK_HIGH_DAYS,
        "peak_window_includes_signal_day": True,
        "peak_drawdown_min_inclusive": P26_PEAK_DRAWDOWN_MIN,
        "additional_indicator_count": 2,
        "additional_indicators": [
            "five-session close return >= 5%",
            "close <= 97% of the highest high in the latest 10 sessions",
        ],
        "primary_horizons": list(VERIFY_HORIZONS),
    }


# --- 试盘 🟡hold ---
def _pat_test_upper_shadow(bars, ctx):           # P6 试盘长上影破平台又缩回
    if len(bars) < 30:
        return False
    closes = ctx["closes"]
    for i in range(len(bars) - 8, len(bars) - 1):
        up, body = _kl_upper(bars, i), _kl_body(bars, i)
        prior = [bars[j]["high"] for j in range(max(0, i - 20), i) if bars[j]["high"] is not None]
        if (up and body and up > 0.03 and up > 2 * body and prior and bars[i]["high"]
                and bars[i]["high"] > max(prior)
                and closes[-1] and closes[i] and closes[-1] < closes[i]):
            return True
    return False


def _pat_bottom_spike(bars, ctx):                # P7 底部异动放量
    pos, vr = ctx["pos"], ctx["vol_ratio"]
    if pos is None or pos >= 0.40 or vr is None or vr <= 1.5 or len(bars) < 6:
        return False
    amps = [a for a in (_kl_amp(bars, i) for i in range(len(bars) - 5, len(bars))) if a is not None]
    return bool(amps and max(amps) > 0.07)


# --- 洗盘 🟡buy ---
def _pat_pullback_shakeout(bars, ctx):           # P8 缩量回踩洗盘(挖坑)
    closes, vol, ma20 = ctx["closes"], ctx["vol"], ctx["ma20"]
    if ma20 is None or len(bars) < 12:
        return False
    c = closes[-1]
    if not c or c < ma20:
        return False
    hi = max([x for x in closes[-10:] if x] or [c])
    if not (-0.15 <= c / hi - 1 <= -0.02):
        return False
    down_v = [vol[i] for i in range(len(bars) - 8, len(bars))
              if bars[i]["chg"] is not None and bars[i]["chg"] < 0 and vol[i]]
    up_v = [vol[i] for i in range(len(bars) - 8, len(bars))
            if bars[i]["chg"] is not None and bars[i]["chg"] > 0 and vol[i]]
    if not down_v or not up_v:
        return False
    chip = _ensure_pattern_chip_context(ctx)
    conc_ok = chip is None or chip["concentration"] >= 0.40
    return (sum(down_v) / len(down_v)) < (sum(up_v) / len(up_v)) and conc_ok


def _pat_climb_wash(bars, ctx):                  # P9 边拉边洗
    if not ctx["ma_bull"] or len(bars) < 12:
        return False
    signs = [1 if (bars[i]["chg"] or 0) > 0 else -1 for i in range(len(bars) - 8, len(bars))]
    alt = sum(1 for k in range(1, len(signs)) if signs[k] != signs[k - 1])
    lows = [bars[i]["low"] for i in range(len(bars) - 10, len(bars)) if bars[i]["low"] is not None]
    higher_low = len(lows) >= 8 and min(lows[-5:]) > min(lows[:5])
    return alt >= 4 and higher_low


def _pat_high_turnover_wash(bars, ctx):          # P10 高换手洗盘(筹码峰不发散)
    vr, ma20, closes = ctx["vol_ratio"], ctx["ma20"], ctx["closes"]
    if not (vr is not None and vr > 1.5 and ma20 and closes[-1] and closes[-1] > ma20):
        return False
    chip = _ensure_pattern_chip_context(ctx)
    return bool(chip is not None and chip["concentration"] >= 0.45)


# --- 突破 🟠hold ---
def _pat_breakout(bars, ctx):                    # P11 放量突破启动(右进左出)
    return bool(ctx["triggered"])


# --- 拉升 🟠hold ---
def _pat_consecutive_limit(bars, ctx):           # P12 连板拉升
    return ctx["streak"] >= 2


def _pat_first_board(bars, ctx):                 # P13 首板卡位
    if len(bars) < 22:
        return False
    limit = _limit_pct(ctx["code"]) - 0.3
    today = bars[-1]["chg"]
    if today is None or today < limit or ctx["streak"] != 1:
        return False
    for i in range(len(bars) - 21, len(bars) - 1):
        if bars[i]["chg"] is not None and bars[i]["chg"] >= limit:
            return False
    t = bars[-1]["turnover"]
    return t is not None and 10.0 <= t <= 45.0


# --- 出货 🔴sell ---
def _pat_high_vol_stall(bars, ctx):              # P14 高位放量滞涨
    pos, vr, closes = ctx["pos"], ctx["vol_ratio"], ctx["closes"]
    if pos is None or pos < 0.85 or vr is None or vr <= 1.5:
        return False
    r5 = _ret_k(closes, 5)
    up = _kl_upper(bars, len(bars) - 1)
    return r5 is not None and r5 <= 0.02 and up is not None and up > 0.02


def _p15_original_event(
    bars: Sequence[Dict[str, Any]],
    vol: Optional[Sequence[Optional[float]]] = None,
) -> bool:
    """复现旧P15的新高量背离状态，仅作为潜在顶部事件而非卖点。"""
    if len(bars) < P15_EVENT_HIGH_DAYS:
        return False
    event_bars = list(bars[-LOOKBACK:])
    if vol is None:
        event_vol, _ = _volume_series(event_bars)
    else:
        if len(vol) != len(bars):
            return False
        event_vol = list(vol[-LOOKBACK:])
    closes = [_safe(bar.get("close")) for bar in event_bars]
    recent = [value for value in closes[-P15_EVENT_HIGH_DAYS:] if value]
    close = closes[-1]
    if len(recent) < 30 or close is None or close < max(recent):
        return False
    _, volume_ratio = _score_volume_ratio(event_vol)
    if volume_ratio is None:
        return False
    ret5 = _ret_k(closes, 5)
    return bool(
        volume_ratio < P15_EVENT_LOW_VOLUME_RATIO_MAX
        or (
            volume_ratio > P15_EVENT_HIGH_VOLUME_RATIO_MIN
            and ret5 is not None
            and ret5 < P15_EVENT_RET5_MAX
        )
    )


def _pat_vol_price_div(bars, ctx):               # P15 新高量背离放量回撤确认
    """旧P15事件后15日内出现放量回撤，才升级为卖出信号。

    旧P15中约99%的命中来自“缩量收盘新高”，单独看更像供给枯竭后的
    趋势延续。因此新规则保留它作为潜在顶部事件，但把信号日后移到
    价格和成交共同确认转弱的当日，避免未来数据回填到事件日。
    """
    # 需要完整保留“当前180日上下文 + 事件回看15日”，否则较早事件日的
    # 换手率覆盖率量源可能只基于截短历史，导致研究与线上口径漂移。
    required = PATTERN_EVAL_BARS
    day_volume_ratio = _safe(ctx.get("day_volume_ratio"))
    if (
        len(bars) < required
        or day_volume_ratio is None
        or day_volume_ratio <= P15_DAY_VOLUME_RATIO_MIN
    ):
        return False

    close = _safe(bars[-1].get("close"))
    recent_highs = [
        _safe(bar.get("high")) for bar in bars[-P15_PULLBACK_HIGH_DAYS:]
    ]
    if (
        close is None
        or close <= 0.0
        or len(recent_highs) != P15_PULLBACK_HIGH_DAYS
        or any(value is None or value <= 0.0 for value in recent_highs)
    ):
        return False
    peak = max(float(value) for value in recent_highs if value is not None)
    if close > peak * (1.0 - P15_PULLBACK_MIN):
        return False

    for lag in range(1, P15_EVENT_LOOKBACK_DAYS + 1):
        event_end = len(bars) - lag
        if _p15_original_event(bars[:event_end]):
            return True
    return False


def _p15_rule_params() -> Dict[str, Any]:
    """P15生产/回测共用的可审计参数快照。"""
    return {
        "code": "P15",
        "name": "新高量背离放量回撤",
        "scope": "leader_and_hotmoney",
        "validated_scope": "leader_and_hotmoney",
        "minimum_history_bars": PATTERN_EVAL_BARS,
        "event_new_close_high_days": P15_EVENT_HIGH_DAYS,
        "event_lookback_days": P15_EVENT_LOOKBACK_DAYS,
        "event_signal_day_excluded": True,
        "event_low_volume_ratio_max_exclusive": P15_EVENT_LOW_VOLUME_RATIO_MAX,
        "event_high_volume_ratio_min_exclusive": P15_EVENT_HIGH_VOLUME_RATIO_MIN,
        "event_high_volume_ret5_max_exclusive": P15_EVENT_RET5_MAX,
        "event_volume_ratio_recent_days": SHORT_WIN,
        "event_volume_ratio_base_days": BASE_WIN,
        "confirmation_peak_high_days": P15_PULLBACK_HIGH_DAYS,
        "confirmation_peak_includes_signal_day": True,
        "confirmation_drawdown_min_inclusive": P15_PULLBACK_MIN,
        "confirmation_day_volume_ratio_min_exclusive": P15_DAY_VOLUME_RATIO_MIN,
        "confirmation_day_volume_base_days": P15_DAY_VOLUME_BASE_DAYS,
        "confirmation_day_volume_min_valid_observations": P15_DAY_VOLUME_MIN_OBS,
        "volume_source": (
            "turnover if trailing-180 PIT coverage >=70%, otherwise volume"
        ),
        "additional_indicator_count": 2,
        "additional_indicators": [
            "signal-day close drawdown from the inclusive 10-session intraday high",
            "signal-day measure divided by the preceding 20-session mean",
        ],
        "event_to_signal_policy": (
            "the original P15 is only a latent event; the sell signal is dated on "
            "the later confirmation day, never backfilled"
        ),
        "p22_overlap_share": {"leader": 0.13715596, "hotmoney": 0.18760234},
        "p15_only_coverage": {"leader": 0.00515987, "hotmoney": 0.00693590},
        "backtest_snapshot": {
            "leader": {
                "hits": 10900,
                "coverage": 0.00598008,
                "win_rates": [0.45706422, 0.45348624, 0.44908257, 0.45495413, 0.43798165],
                "excess_returns": [-0.00382091, -0.00485528, -0.00682607, -0.00728918, -0.01138214],
                "effective_at": list(VERIFY_HORIZONS),
            },
            "hotmoney": {
                "hits": 4275,
                "coverage": 0.00853757,
                "win_rates": [0.40187135, 0.38877193, 0.37052632, 0.37754386, 0.39111111],
                "excess_returns": [-0.01095574, -0.01885232, -0.02974947, -0.03924387, -0.03942149],
                "effective_at": list(VERIFY_HORIZONS),
            },
        },
    }


def _pat_bearish_max_vol(bars, ctx):             # P16 阴天量回撤确认
    pos, vol = ctx["pos"], ctx["vol"]
    if (
        pos is None
        or pos < P16_POSITION_MIN
        or len(bars) < P16_MAX_VOLUME_DAYS
        or len(vol) < P16_MAX_VOLUME_DAYS
    ):
        return False

    # 40日最大量只说明出现了高位分歧；当前收盘较近10日盘中高点至少回撤3%，
    # 才确认分歧已经演变为价格走弱。高点窗口含当前日，允许识别当日冲高回落。
    close = _safe(bars[-1].get("close"))
    recent_highs = [
        value
        for value in (_safe(bar.get("high")) for bar in bars[-P16_PEAK_HIGH_DAYS:])
        if value is not None and value > 0.0
    ]
    if (
        close is None
        or close <= 0.0
        or not recent_highs
        or close > (1.0 - P16_PEAK_DRAWDOWN_MIN) * max(recent_highs)
    ):
        return False

    mx = max([v for v in vol[-P16_MAX_VOLUME_DAYS:] if v] or [0])
    for i in range(len(bars) - P16_EVENT_RECENCY_DAYS, len(bars)):
        if vol[i] and vol[i] >= mx and bars[i]["chg"] is not None and bars[i]["chg"] < 0:
            return True
    return False


def _p16_rule_params() -> Dict[str, Any]:
    """P16 冻结生产口径，供回测结果、接口和测试审计。"""
    return {
        "scope": "leader_and_hotmoney",
        "position_min_inclusive": P16_POSITION_MIN,
        "max_volume_days": P16_MAX_VOLUME_DAYS,
        "bearish_event_recency_days": P16_EVENT_RECENCY_DAYS,
        "bearish_event_change_max_exclusive": 0.0,
        "peak_high_days": P16_PEAK_HIGH_DAYS,
        "peak_drawdown_min_inclusive": P16_PEAK_DRAWDOWN_MIN,
        "peak_window_includes_signal_day": True,
        "additional_indicator_count": 1,
        "additional_indicator": "close <= 97% of the highest high in the latest 10 sessions",
    }


def _pat_inverted_v(bars, ctx):                  # P17 倒V反转
    pos, closes = ctx["pos"], ctx["closes"]
    required = P17_RECENT_DAYS + P17_BASE_DAYS
    if (
        pos is None
        or pos < P17_POSITION_MIN
        or len(bars) < required
        or len(closes) < required
    ):
        return False
    recent_window = closes[-P17_RECENT_DAYS:]
    recent = [c for c in recent_window if c]
    base = [c for c in closes[-required:-P17_RECENT_DAYS] if c]
    if len(recent) < 8 or not base:
        return False
    peak = max(recent)
    last, previous = closes[-1], closes[-2]
    if not last or not previous or not peak:
        return False
    first_peak_index = next(
        index for index, close in enumerate(recent_window) if close == peak
    )
    return bool(
        peak / min(base) - 1 > P17_RUNUP_MIN
        and last / peak - 1 <= -P17_PULLBACK_MIN
        and first_peak_index >= P17_RECENT_DAYS - P17_PEAK_RECENCY_BARS
        and last <= previous
    )


def _p17_rule_params() -> Dict[str, Any]:
    """P17 冻结生产口径，供回测结果、接口和测试审计。"""
    return {
        "scope": "leader_and_hotmoney",
        "position60_min_inclusive": P17_POSITION_MIN,
        "recent_peak_days": P17_RECENT_DAYS,
        "base_days": P17_BASE_DAYS,
        "runup_min_exclusive": P17_RUNUP_MIN,
        "pullback_from_peak_min_inclusive": P17_PULLBACK_MIN,
        "peak_first_seen_within_bars": P17_PEAK_RECENCY_BARS,
        "signal_day_close_not_above_previous": True,
        "additional_requirement_count": 2,
        "additional_requirements": [
            "the first occurrence of the 10-day peak is within the latest 5 bars",
            "signal-day close is not above the previous close",
        ],
    }


def _pat_bearish_engulf(bars, ctx):              # P18 顶部大阴包阳
    if ctx["pos"] is None or ctx["pos"] < 0.80 or len(bars) < 2:
        return False
    o1, c1 = bars[-2]["open"], bars[-2]["close"]
    o0, c0 = bars[-1]["open"], bars[-1]["close"]
    if None in (o1, c1, o0, c0):
        return False
    return c1 > o1 and c0 < o0 and o0 >= c1 and c0 <= o1


def _pat_dump_bigbear(bars, ctx):                # P19 灌压出货(巨量大阴)
    pos, vr = ctx["pos"], ctx["vol_ratio"]
    if pos is None or pos < 0.70 or vr is None or vr <= 1.8:
        return False
    body = _kl_body(bars, len(bars) - 1)
    o, c, h, l = bars[-1]["open"], bars[-1]["close"], bars[-1]["high"], bars[-1]["low"]
    if None in (o, c, h, l) or h <= l:
        return False
    return body is not None and body > 0.06 and c < o and (c - l) / (h - l) < 0.25


def _ma_breakdown(closes: List[Optional[float]], vol_ratio: Optional[float]) -> bool:
    """P20：跌破MA20 + 第5根bar仍在MA20上方 + 放量 + 当日确认下跌。"""
    ma20 = _ma_last(closes, 20)
    if ma20 is None or len(closes) < 25 or vol_ratio is None:
        return False
    c, previous = closes[-1], closes[-2]
    if not c or not previous or c >= ma20:
        return False
    prior, ma20_prior = closes[-5], _ma_last(closes[:-4], 20)
    day_return = c / previous - 1.0
    return bool(
        prior
        and ma20_prior
        and prior > ma20_prior
        and vol_ratio > P20_VOLUME_RATIO_MIN
        and day_return <= P20_DAY_RETURN_MAX
    )


def _pat_ma_breakdown(bars, ctx):                # P20 均线放量破位
    return _ma_breakdown(ctx["closes"], ctx["vol_ratio"])


def _p20_rule_params() -> Dict[str, Any]:
    """P20 冻结生产口径，供回测结果和测试审计。"""
    return {
        "scope": "leader_and_hotmoney",
        "ma_days": 20,
        "prior_bar_index": -5,
        "prior_sessions_ago": 4,
        "volume_ratio_recent_days": SHORT_WIN,
        "volume_ratio_base_days": BASE_WIN,
        "volume_ratio_min_exclusive": P20_VOLUME_RATIO_MIN,
        "day_return_max_inclusive": P20_DAY_RETURN_MAX,
        "incremental_requirement_count": 1,
        "incremental_requirement": "signal-day close-to-close return <= -1%",
    }


def _p21_spring_state(bars: List[Dict[str, Any]]) -> Optional[bool]:
    """P21价格主体：近5日深破前40日箱底至少2.5%，当前收盘重新站回。

    固定按交易日切片，不跳过缺失 low 后再错位补数；数据不足或窗口内有脏值时返回
    ``None`` 并关闭信号。该主体刻意保留为最多5日的有效状态，而不是独立事件：专项
    复测中 False→True 事件化会把双池覆盖压到0.5%以下，且近期小盘超额转负。
    """
    required = P21_BOX_DAYS + P21_RECLAIM_DAYS
    if len(bars) < required:
        return None
    window = bars[-required:]
    box_lows = [_safe(bar.get("low")) for bar in window[:P21_BOX_DAYS]]
    recent_lows = [_safe(bar.get("low")) for bar in window[P21_BOX_DAYS:]]
    close = _safe(window[-1].get("close"))
    if (
        close is None
        or close <= 0.0
        or any(value is None or value <= 0.0 for value in box_lows + recent_lows)
    ):
        return None
    numeric_box_lows = [float(value) for value in box_lows if value is not None]
    numeric_recent_lows = [float(value) for value in recent_lows if value is not None]
    box_low = min(numeric_box_lows)
    recent_low = min(numeric_recent_lows)
    return bool(
        recent_low < box_low * P21_UNDERCUT_RATIO_MAX
        and close > box_low
    )


def _p21_temperate_volume_state(
    bars: List[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """P21供给枯竭确认：信号日量能不高于此前20日均值的75%。"""
    if ctx is not None and "p26_volume_ratio" in ctx:
        day_volume_ratio = ctx.get("p26_volume_ratio")
    else:
        current_bars = bars[-LOOKBACK:]
        volume, _source = _volume_series(current_bars)
        day_volume_ratio = _p26_volume_ratio(volume)
    if day_volume_ratio is None:
        return None
    return bool(day_volume_ratio <= P21_DAY_VOLUME_RATIO_MAX)


def _pat_spring_reclaim(bars, ctx):              # P21 深破缩量收回(Wyckoff spring)
    """深破2.5%后站回箱底，并以不超过0.75倍量确认抛压枯竭。"""
    if len(bars) < P21_MIN_HISTORY_BARS:
        return False
    return bool(
        _p21_spring_state(bars) is True
        and _p21_temperate_volume_state(bars, ctx=ctx) is True
    )


def _p21_rule_params() -> Dict[str, Any]:
    """P21生产/回测共用的可审计参数快照。"""
    return {
        "code": "P21",
        "name": "深破缩量收回",
        "scope": "leader_and_hotmoney",
        "validated_scope": "leader_and_hotmoney",
        "primary_horizons": [10, 20, 40],
        "minimum_history_bars": P21_MIN_HISTORY_BARS,
        "box_days": P21_BOX_DAYS,
        "reclaim_window_days": P21_RECLAIM_DAYS,
        "undercut_ratio_max_exclusive": P21_UNDERCUT_RATIO_MAX,
        "minimum_undercut_pct_exclusive": round(
            1.0 - P21_UNDERCUT_RATIO_MAX, 6
        ),
        "close_must_reclaim_box_low": True,
        "day_volume_ratio_max_inclusive": P21_DAY_VOLUME_RATIO_MAX,
        "day_volume_ratio_base_days": CHIP_WINNER_RISK_VOLUME_BASE_WIN,
        "day_volume_source": (
            "turnover if trailing-180 PIT coverage >=70%, otherwise volume"
        ),
        "additional_indicator_count": 1,
        "additional_indicator": "signal-day volume / prior-20-day mean",
        "state_rule": (
            "daily while the latest 5-day window still contains the qualifying "
            "undercut, close remains above the box low, and signal-day volume passes"
        ),
        "eventized": False,
        "eventized_rejection": (
            "False-to-True coverage was 0.2585%/0.4051% in leader/hotmoney, "
            "below the 0.5% product floor"
        ),
        "maximum_undercut": None,
        "maximum_undercut_rejection": (
            "a 5% depth cap reduced leader coverage below 0.5% and weakened "
            "development results"
        ),
        "backtest_snapshot": {
            "leader": {
                "hits": 11402,
                "coverage": 0.006236,
                "win_rate_10d": 0.5453,
                "excess_10d": 0.00332,
                "win_rate_20d": 0.5644,
                "excess_20d": 0.00510,
            },
            "hotmoney": {
                "hits": 4566,
                "coverage": 0.009125,
                "win_rate_10d": 0.5644,
                "excess_10d": 0.00421,
                "win_rate_20d": 0.6378,
                "excess_20d": 0.01327,
            },
        },
    }


def _pat_failed_breakout(bars, ctx):             # P22 放量假突破(高位拒绝)
    """今日盘中高点刺破前40日高点、但收盘没站上 + 当日放量(>1.8×前20日均量)。沿用早期形态研究。"""
    if len(bars) < 42:
        return False
    highs = [b["high"] for b in bars if b["high"] is not None]
    if len(highs) < 42:
        return False
    high_40_prior = max(highs[-41:-1])           # 前40日(不含今日)最高
    today_high, c = bars[-1]["high"], ctx["closes"][-1]
    vol = ctx["vol"]
    base = _mean([v for v in vol[-21:-1] if v])
    today_v = vol[-1] if vol else None
    if not high_40_prior or today_high is None or not c or not base or not today_v:
        return False
    day_ratio = today_v / base
    return today_high > high_40_prior and c <= high_40_prior and day_ratio > 1.8


# (code, 名称, 阶段, 信号方向, 匹配函数)
PATTERNS: List[Tuple[str, str, str, str, Any]] = [
    ("P1", "超额优先复合确认", "吸筹", "buy", _pat_low_consolidation),
    ("P2", "低位影线吸筹", "吸筹", "buy", _pat_low_shadows),
    ("P3", "缩量打压首次收复", "吸筹", "buy", _pat_shakedown_absorb),
    ("P4", "量增价稳观察", "吸筹", "observe", _pat_absorption),
    ("P5", "双底右腿确认", "吸筹", "buy", _pat_bottom_formation),
    ("P6", "试盘长上影", "试盘", "hold", _pat_test_upper_shadow),
    ("P7", "底部异动放量", "试盘", "hold", _pat_bottom_spike),
    ("P8", "缩量回踩洗盘", "洗盘", "buy", _pat_pullback_shakeout),
    ("P9", "边拉边洗", "洗盘", "buy", _pat_climb_wash),
    ("P10", "高换手洗盘", "洗盘", "buy", _pat_high_turnover_wash),
    ("P11", "放量突破启动", "突破", "hold", _pat_breakout),
    ("P12", "连板拉升", "拉升", "hold", _pat_consecutive_limit),
    ("P13", "首板卡位", "拉升", "hold", _pat_first_board),
    ("P14", "高位放量滞涨", "出货", "sell", _pat_high_vol_stall),
    ("P15", "新高量背离放量回撤", "出货", "sell", _pat_vol_price_div),
    ("P16", "阴天量", "出货", "sell", _pat_bearish_max_vol),
    ("P17", "倒V反转", "出货", "sell", _pat_inverted_v),
    ("P18", "顶部大阴包阳", "出货", "sell", _pat_bearish_engulf),
    ("P19", "灌压巨量大阴", "出货", "sell", _pat_dump_bigbear),
    ("P20", "均线放量破位", "出货", "sell", _pat_ma_breakdown),
    ("P21", "深破缩量收回", "洗盘", "buy", _pat_spring_reclaim),     # 双池10~40日正向，核心洗盘买点
    ("P22", "放量假突破", "出货", "sell", _pat_failed_breakout),     # 全历史逐日复测：双池2~40日风控有效
    ("P23", "低位压缩温和量能转强", "吸筹", "buy", _pat_compression),  # 10日正向，核心有效买点
    ("P24", "OBV底背离", "吸筹", "buy", _pat_obv_divergence),        # 强OBV背离，连续5日后一次性确认
    ("P25", "中低位缩量平台转强", "吸筹", "buy", _pat_bottom_base_ignition),
    ("P26", "获利盘风险", "出货", "sell", _pat_chip_winner_risk),
]

# 每个形态的命中条件一句话（供前端「命中形态解释」模块；与文件头总表口径一致）。
PATTERN_DESC: Dict[str, str] = {
    "P1": "原低位缩量企稳两日首次确认，或低位首次突破前10日最高收盘并通过510310日终风险门；当前全历史复测覆盖率leader/hotmoney 0.5185%/0.5085%，10日胜率54.79%/57.56%、同池超额+0.464%/+1.297%、HAC t=3.21/3.24；小盘更有效（hotmoney池效果更强），进入吸筹分10%",
    "P2": "位置<0.40 + 近10日真实长下影≥2根(下影≥2×实体且≥2%) + 当日量能/此前20日均量0.4~1.0：温和缩量探底承接；全历史复测双池10/20日正超额；大盘更有效（leader池效果更强），进入吸筹分5%",
    "P3": "位置<0.32 + 近10日收盘跌≥4.2%且量低于此前20日均量，事件后收盘最多再破底6%，确认量比≤1.35并真正首次完整收复；全历史复测双池命中率0.1005%/0.2002%，2~40日超额均为正",
    "P4": "位置<0.60 + 近5日/前20日量比1.2~2.5 + |20日涨跌|<6% + CMF>0，仅在核心状态首次进入时报告；覆盖率约1.17%/1.06%，双池留出未验证为买点，保留中性结构观察且不驱动绿色吸筹阶段",
    "P5": "位置<0.45 + 两摆动低点(-2%~+10%) + 颈线反弹≥5% + 二底后8日内；今日触及颈线且由昨日未突破切换为突破前3日收盘高点。双池全历史复测命中率0.116%/0.131%，10日胜率55.2%/63.8%、同池超额+0.35%/+0.84%；小盘更有效（hotmoney池效果更强），按产品决定暂时纳入核心有效集并进入吸筹分5%",
    "P6": "近8日长上影(>3%且>2×实体)创20日新高后收盘缩回：探顶又压回；保留形态观测，不作为有效因子",
    "P7": "位置<0.40 + 量比>1.5 + 近5日最大振幅>7%：低位突然放量异动；保留形态观测，不作为有效因子",
    "P8": "站上MA20 + 回踩近10日高点-2%~-15% + 跌日量<涨日量 + 筹码集中≥0.40：挖坑不破位",
    "P9": "多头排列(MA5>10>20) + 近8日涨跌切换≥4次 + 低点抬高：边拉边洗",
    "P10": "量比>1.5 + 收盘>MA20 + 筹码集中≥0.45：高换手震仓但筹码峰不发散",
    "P11": "右进左出触发器：从低/中位放量(>1.3×)突破近20日收盘高点；全历史逐日复测为双池负向风险",
    "P12": "连续涨停 streak≥2；全历史逐日复测：2日动量有效，hotmoney 池10~40日转为反转风险",
    "P13": "今日首板(近20日无涨停) + 换手10%~45%；全历史逐日复测：2日动量有效，20~40日转为反转风险",
    "P14": "位置≥0.85 + 量比>1.5 + 近5日涨幅≤2% + 上影>2%：高位放量不涨；全历史逐日复测：双池2~40日风控有效",
    "P15": "前15日内曾出现旧P15新高量背离，当前收盘较含当日的近10日盘中高点回撤至少4%，且当日量能/此前20日均量>1.5；信号记在确认日。全历史复测命中率leader/hotmoney为0.598%/0.854%，双池2~40日胜率<50%且同池超额均为负",
    "P16": "位置≥0.80 + 近3日出现40日最大量且相对昨收下跌 + 当前收盘较近10日最高价回撤≥3%：天量分歧后的价格走弱确认；全历史复测双池命中率0.596%/0.926%，2~40日胜率与同池超额均低于旧版",
    "P17": "位置≥0.80 + 较前期冲高>16%后从峰值回落≤-8%，且峰值首次出现于最近5根、信号日不反弹：新鲜倒V；全历史复测双池命中率约0.67%/1.41%，2~40日胜率与超额均低于旧版",
    "P18": "位置≥0.80 + 昨阳今阴且今实体完全吞没昨实体：顶部看跌吞没",
    "P19": "位置≥0.70 + 量比>1.8 + 实体跌>6%且收在当日价区下1/4：巨量灌压大阴；全历史逐日复测：双池2~40日持续负超额",
    "P20": "收盘跌破MA20(第5根bar仍在当时MA20上方) + 近5/此前20日量能比>1.3 + 当日跌≥1%：确认放量破位；全历史复测双池命中率1.08%/1.41%，2~40日胜率<50%且负超额",
    "P21": "近5日最低价深破前40日箱底至少2.5%，当前收盘重新站回，且信号日量能不高于此前20日均值的0.75倍：供给枯竭后的Wyckoff spring；全历史复测双池10/20/40日胜率与同池超额均为正，进入吸筹分10%",
    "P22": "今日盘中破前40日高但收盘没站上 + 当日放量>1.8×：放量假突破(高位拒绝)；全历史逐日复测：双池2~40日风控有效",
    "P23": "位置<0.36 + 近20/60日振幅比<0.96 + 收盘≥1.005×MA20，只在该结构状态首次成立且信号日量能/此前20日均量在0.4~1.2时触发；冻结快照生产等价复测覆盖率leader/hotmoney 0.678%/0.749%，10日胜率51.83%/52.19%、同池超额+0.331%/+0.529%；按产品口径纳入核心有效集并进入吸筹分5%",
    "P24": "位置<0.50 + 30日涨幅≤3% + 量纲化OBV>0.10 + OBV价差>0.25，连续5日后仅首次确认；最新复测双池10日有效",
    "P25": "双池计算并进入吸筹分5%：120日价格分位≤65% + 60日振幅≤35%且涨跌≤16% + 近20日/前40日均量≤70% + 量能斜率≤-0.5% + 20日涨幅≥0.5% + 距20日平台≤4%；全历史复测命中率leader/hotmoney 1.326%/1.254%，10日胜率52.4%/53.7%、同池超额+0.24%/+0.55%，方向跨早晚段为正但leader HAC未显著；按产品口径纳入核心有效集",
    "P26": "获利盘比例≥90%，或5个交易日前<40%且当前≥60%；同时要求当日量能≥此前20日均量2倍、近5日涨幅≥5%、当前收盘较近10日盘中高点回撤≥3%：只把急涨后的极端放量回撤升级为兑现风险。全历史复测命中率leader/hotmoney 0.631%/0.995%，双池2~40日胜率均<50%且同池超额均为负；leader 40日HAC尚未显著",
}

# 当前生产核心有效形态（前端高亮）：P1/P2按2026-07-13双池复测与产品决定加入；
#   P12/P13 为超短动量，P1/P2/P3/P5/P21/P23/P24/P25 为吸筹/洗盘买点，P26为获利盘兑现风险，
#   其余为负向风险/出货风控；P1/P3/P21各占吸筹分10%，
#   P2/P5/P23/P24/P25各占5%。
PATTERN_EFFECTIVE = {"P1", "P2", "P3", "P5", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P19", "P20", "P21", "P22", "P23", "P24", "P25", "P26"}
# “有效卖出形态”取核心有效集与结构 sell 信号的交集。P11 虽为统计负向
# risk 样式，但原始结构是 hold；P18 仍只作形态观察。
PATTERN_EFFECTIVE_SELL = frozenset(
    code
    for code, _name, _phase, signal, _matcher in PATTERNS
    if code in PATTERN_EFFECTIVE and signal == "sell"
)
PATTERN_EFFECTIVE_STYLE = {
    "P1": "bullish",
    "P2": "bullish",
    "P3": "bullish",
    "P5": "bullish",
    "P11": "risk",
    "P12": "momentum",
    "P13": "momentum",
    "P14": "risk",
    "P15": "risk",
    "P16": "risk",
    "P17": "risk",
    "P19": "risk",
    "P20": "risk",
    "P21": "bullish",
    "P22": "risk",
    "P23": "bullish",
    "P24": "bullish",
    "P25": "bullish",
    "P26": "risk",
}


def distribution_warning_rule_metadata() -> Dict[str, Any]:
    """返回列表与形态回测共用的出货预警触发口径。"""
    return {
        "type": "any_of",
        "operator": "or",
        "pattern_codes": sorted(DISTRIBUTION_WARNING_PATTERN_CODES),
        "trigger_on_any_pattern": True,
        "effective_pattern_count_threshold": (
            DISTRIBUTION_WARNING_EFFECTIVE_SELL_THRESHOLD
        ),
        "points_threshold": DISTRIBUTION_WARNING_POINTS_THRESHOLD,
        "conditions": [
            {
                "type": "effective_sell_pattern_count",
                "operator": "gte",
                "threshold": DISTRIBUTION_WARNING_EFFECTIVE_SELL_THRESHOLD,
            },
            {
                "type": "distribution_warning_points",
                "operator": "gte",
                "threshold": DISTRIBUTION_WARNING_POINTS_THRESHOLD,
            },
        ],
        "points_explanation_only": False,
    }


def pattern_catalog() -> List[Dict[str, Any]]:
    """形态总表结构化输出，含后端统一维护的有效性与前端颜色语义。"""
    out: List[Dict[str, Any]] = []
    for code, name, category, signal, _ in PATTERNS:
        key = code.lower()
        if key in ACCUM_MODEL_WEIGHTS:
            score_usage = f"吸筹分 {round(ACCUM_MODEL_WEIGHTS[key] * 100)}%"
        elif key in DIST_MODEL_WEIGHTS:
            score_usage = f"出货分 {round(DIST_MODEL_WEIGHTS[key] * 100)}%"
        elif code in DISTRIBUTION_WARNING_PATTERN_CODES:
            score_usage = "出货预警"
        else:
            score_usage = "阶段标签"

        if code in PATTERN_EFFECTIVE:
            validation_status, validation_label = "core", "核心有效"
        else:
            validation_status, validation_label = "observation", "形态观察"

        # 展示颜色表达生产信号方向；有效性与验证标签仍由独立字段明确输出。
        display_style = (
            "bullish"
            if code in PATTERN_BACKTEST_BUY
            else PATTERN_EFFECTIVE_STYLE.get(code, "neutral")
        )
        out.append({
            "code": code, "name": name, "category": category, "signal": signal,
            "desc": PATTERN_DESC.get(code, ""), "effective": code in PATTERN_EFFECTIVE,
            "effective_style": PATTERN_EFFECTIVE_STYLE.get(code, "neutral"),
            "display_style": display_style,
            "production": True, "score_usage": score_usage,
            "validation_status": validation_status, "validation_label": validation_label,
        })
    return out


def scoring_model_catalog() -> Dict[str, Any]:
    """前端说明弹窗的唯一模型口径；权重直接来自生产常量，避免文案漂移。"""
    accumulation = {
        "chip": ("筹码集中", "低位筹码峰越集中、当前价越接近主要成本区，分数越高。"),
        "position": ("价格位置", "当前价格在近60日区间越靠下，分数越高；它偏向低位反转。"),
        "cmf_eff": ("CMF反向分", "高买压在历史中更像短期过热，因此只惩罚过高买压。"),
        "p2": ("P2 低位长下影承接", "低位反复出现真实长下影，并由温和缩量确认；大盘更有效，命中按5分计入。"),
        "p3": ("P3 缩量打压首次收复", "低位缩量收盘大跌后，事件后从未站回且收盘最多再破底6%，当日首次完整收回下跌前收盘。"),
        "p5": ("P5 双底右腿确认", "二底形成后回到颈线，并首次突破近3日收盘高点；小盘更有效，按产品决定暂时纳入核心有效集，命中计5分。"),
        "p21": ("P21 深破缩量收回", "价格深破前40日箱底后重新站回，且信号日缩量；命中按10分计入。"),
        "p23": ("P23 低位压缩温和量能转强", "低位波动压缩后站上MA20，并由温和量能确认；核心有效，命中按5分计入。"),
        "p24": ("P24 OBV底背离", "价格仍在低位，量纲化OBV持续走强并连续确认后首次触发。"),
        "p1": ("P1 超额优先复合确认", "原低位缩量企稳两日首次确认，或低位首次突破前10日最高收盘并通过510310日终风险门；小盘更有效，双池命中计10分。"),
        "p25": ("P25 缩量平台转强", "双池命中均计5分；中低位缩量平台的20日价格开始转正并接近平台。核心有效；双池10/20日胜率和同池超额方向为正，但leader尚未达到HAC显著。"),
        "holder_change": ("股东户数变化", "户数下降得分更高；-15%映射100分、+15%映射0分，缺失按50分。"),
        "repurchase": ("近90日回购", "近90个自然日有回购公告得100分，否则0分。"),
    }
    distribution = {
        "p14": ("P14 高位放量滞涨", "处于高位、明显放量，但价格不再有效上涨。"),
        "p15": ("P15 新高量背离放量回撤", "前15日出现新高量背离事件后，当前放量并从近10日盘中高点明显回撤，确认顶部风险。"),
        "p16": ("P16 阴天量回撤确认", "高位出现阶段最大量收跌后，仅在3日时效内且价格已从近10日高点回撤至少3%时确认。"),
        "p17": ("P17 新鲜倒V反转", "高位冲高后从新近峰值快速回落，且信号日继续走弱，过滤陈旧或反弹中的倒V状态。"),
        "p19": ("P19 灌压巨量大阴", "高位巨量长阴且收在日内低位，反映集中抛压。"),
        "p20": ("P20 均线放量破位", "此前位于MA20上方，随后放量跌破均线。"),
        "p22": ("P22 放量假突破", "盘中突破前高但收盘未站稳，同时明显放量。"),
        "p26": ("P26 获利盘冲高回撤风险", "获利盘拥挤或快速转盈后出现极端放量；近5日已有快速拉升、当前又从近10日盘中高点回撤至少3%，确认强势延续已转为兑现风险。"),
        "lhb_recent": ("近90日龙虎榜", "近期上榜常伴随异动和交易拥挤，提示波动加剧。"),
        "technical": ("连续技术派发", "高位门控后，综合20日涨幅和高位放量计算连续风险分。"),
        "divergence": ("原始量价背离", "成交投入大而价格反馈弱，说明资金推动效率下降。"),
    }
    reversal = {
        "turn_pctile": ("换手分位", "越拥挤，反转分越低。"),
        "amp_today": ("当日振幅", "波动越剧烈，反转分越低。"),
        "limitup_5d": ("近5日涨停", "连续过热越明显，反转分越低。"),
        "vol_ratio": ("量比", "突发放量越强，反转分越低。"),
        "mom_5d": ("近5日动量", "近期追涨越强，反转分越低。"),
    }
    auxiliary = [
        {
            "key": "pattern_phase",
            "label": "阶段标签与把握度",
            "description": "P1–P26先决定吸筹、试盘、洗盘、突破、拉升或出货阶段；买入把握沿用形态强度、吸筹分与出货冲突分，出货把握采用PIT时间留出实验校准后的吸筹分/连续出货分概率。",
        },
        {
            "key": "market_regime",
            "label": "大盘状态",
            "description": "游资小盘池读取沪深300ETF累计净值是否站上MA20，只用于判断反转策略是否适合做多，不改变反转分。",
        },
        {
            "key": "industry_heat",
            "label": "二级行业与题材热度",
            "description": "当前雷达用于行业展示和筛选，不进入机会分或反转分；仅在独立潜伏模式中参与潜伏分。",
        },
        {
            "key": "evidence",
            "label": "依据标签",
            "description": "把价格位置、筹码、量价背离、CMF、波动压缩、换手拥挤和形态命中翻译为易读标签；它是解释层，不重复加分。",
        },
    ]

    def rows(weights: Dict[str, float], definitions: Dict[str, Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [
            {"key": key, "label": definitions[key][0], "description": definitions[key][1],
             "weight": float(weight), "weight_pct": round(float(weight) * 100)}
            for key, weight in weights.items()
        ]

    return {
        "opportunity": {
            "formula": OPPORTUNITY_FORMULA,
            "display_formula": "机会分 = 吸筹百分位 ×（1 − 0.5 × 出货百分位 ÷ 100）",
            "distribution_penalty": OPPORTUNITY_DISTRIBUTION_PENALTY,
            "description": "吸筹原始分与出货原始分分别在当前候选池内转换为0–100横截面百分位；同分共享平均名次。出货最多折掉50%的吸筹强度。",
            "example": {"accumulation_percentile": 90, "distribution_percentile": 40, "opportunity_score": 72},
        },
        "accumulation": {
            "formula": "原始特征分 × 权重后求和",
            "factors": rows(ACCUM_MODEL_WEIGHTS, accumulation),
            "suspect_threshold": SUSPECT_ACCUM_SCORE,
            "suspect_rule": "完全未命中任何形态且吸筹分达到阈值",
        },
        "distribution": {
            "formula": "特征值(0–1) × 权重 × 100后求和",
            "factors": rows(DIST_MODEL_WEIGHTS, distribution),
            "warning_rule": distribution_warning_rule_metadata(),
        },
        "reversal": {"formula": "过热因子横截面排名后反向加权，再做近3日EMA", "smooth_days": REVERSAL_SMOOTH_DAYS,
                     "factors": rows(REVERSAL_WEIGHTS, reversal)},
        "auxiliary": auxiliary,
        "sorting": {
            "leader": "细分龙头池按机会分排序",
            "hotmoney": "游资小盘池按反转分排序，机会分仅辅助观察",
            "etf": "ETF 池按技术机会分排序；公司行为因子不适用",
        },
        "pool_overrides": {
            "etf": {
                "accumulation_weights": accumulation_model_weights("etf"),
                "distribution_weights": distribution_model_weights("etf"),
                "excluded_factors": list(ETF_ACCUM_EXCLUDED + ETF_DIST_EXCLUDED),
                "validation": "technical_heuristic_unvalidated_for_etf",
            }
        },
    }


def match_patterns(code: str, bars: List[Dict[str, Any]],
                   ctx: Optional[Dict[str, Any]] = None,
                   pool: Optional[str] = None,
                   p1_market_gate: bool = False,
                   p1_override: Optional[bool] = None,
                   patterns: Optional[Sequence[Tuple[str, str, str, str, Any]]] = None,
                   ) -> List[Dict[str, str]]:
    """对一段日线窗口匹配全部形态，返回命中列表（PIT 安全）。"""
    if len(bars) < MIN_BARS:
        return []
    ctx = ctx or _build_pattern_context(code, bars)
    current_bars = bars[-LOOKBACK:]
    fired: List[Dict[str, str]] = []
    selected_patterns = PATTERNS if patterns is None else patterns
    for pcode, name, phase, signal, fn in selected_patterns:
        # 事件型P1/P4需要让昨日状态也能使用完整180日量度来源窗口；P15
        # 需要按过去15日各事件当日的量源复算；P21额外保留正式回测相同的
        # 185根资格窗口；其余形态只看统一LOOKBACK窗口。
        if pcode == "P1":
            pattern_bars = bars[-(LOOKBACK + P1_CONFIRM_DAYS):]
        elif pcode == "P4":
            pattern_bars = bars[-(LOOKBACK + 1):]
        elif pcode == "P15":
            pattern_bars = bars[-(LOOKBACK + P15_EVENT_LOOKBACK_DAYS):]
        elif pcode == "P21":
            pattern_bars = bars[-P21_MIN_HISTORY_BARS:]
        else:
            pattern_bars = current_bars
        try:
            if pcode == "P1" and fn is _pat_low_consolidation:
                matched = (
                    bool(p1_override)
                    if p1_override is not None
                    else fn(
                        pattern_bars,
                        ctx,
                        p1_market_gate if pool != "etf" else False,
                    )
                )
            else:
                matched = fn(pattern_bars, ctx)
            if matched:
                fired.append({"code": pcode, "name": name, "phase": phase, "signal": signal})
        except Exception:
            continue
    return fired


def _distribution_warning_points(fired: Sequence[Dict[str, str]]) -> int:
    """命中有效出货证据的风险积分；按形态代码去重。"""
    codes = {str(pattern.get("code") or "") for pattern in fired}
    return sum(DISTRIBUTION_WARNING_POINTS.get(code, 0) for code in codes)


def _effective_sell_pattern_count(fired: Sequence[Dict[str, str]]) -> int:
    """命中的出货预警形态数；按代码去重（兼容既有字段名）。"""
    codes = {str(pattern.get("code") or "") for pattern in fired}
    return len(codes & DISTRIBUTION_WARNING_PATTERN_CODES)


def _distribution_warning_triggered(fired: Sequence[Dict[str, str]]) -> bool:
    """统一出货预警规则：预警集合内任一形态命中即触发。"""
    return (
        _effective_sell_pattern_count(fired)
        >= DISTRIBUTION_WARNING_EFFECTIVE_SELL_THRESHOLD
        or _distribution_warning_points(fired)
        >= DISTRIBUTION_WARNING_POINTS_THRESHOLD
    )


def _pattern_phase(fired: List[Dict[str, str]], score: Optional[float] = None) -> str:
    """命中形态汇总成主导阶段（优先级：出货预警达标 > 突破 > 买入区 > 拉升 > 试盘）。

    出货预警只统计 DISTRIBUTION_WARNING_PATTERN_CODES：任一形态命中即触发。
    P11 是 hold 风险，P18 不在预警集合内，不会单独触发。
    突破=放量右进买点，最 actionable，仅次于达标的出货风控示警、优先于被动的吸筹/洗盘；
    买入区按类别细分吸筹 / 洗盘（两类都中则合并标注）；剩余 hold 区拉升中 > 试盘；
    无任何形态命中：吸筹分≥SUSPECT_ACCUM_SCORE → 疑似吸筹(待确认)，否则 → 观望(场外不参与)。
    """
    sigs = {p["signal"] for p in fired}
    cats = {p["phase"] for p in fired}
    if _distribution_warning_triggered(fired):
        return "出货预警🔴"
    if "突破" in cats:
        return "▲突破🟠"
    if "buy" in sigs:
        buy_cats = {p["phase"] for p in fired if p["signal"] == "buy"}
        if {"吸筹", "洗盘"} <= buy_cats:
            return "吸筹+洗盘🟡"
        if "洗盘" in buy_cats:
            return "洗盘🟡"
        return "吸筹🟢"
    if "hold" in sigs:                # 此时仅剩 拉升 / 试盘
        return "拉升中🟠" if "拉升" in cats else "试盘🟡"
    if not fired and score is not None and score >= SUSPECT_ACCUM_SCORE:
        return "疑似吸筹(待确认)🟢"
    return "观望⚪"


def _score_distribution(close_pctile: Optional[float], drift: Optional[float],
                        vol_ratio: Optional[float]) -> float:
    """高位派发风险（连续 0~100）：高位门控 ×（基础高位 + 阶段涨幅 + 高位放量）。

    只有处在高位（close_pctile≥DIST_POS_START）才计派发风险——低位放量是吸筹、高位放量才是出货
    （"天量见天价"）。用于给吸筹分做渐变折扣，而非靠形态硬覆盖标签。
    """
    if close_pctile is None:
        return 0.0
    high = _clip01((close_pctile - DIST_POS_START) / (DIST_POS_FULL - DIST_POS_START))
    if high <= 0.0:
        return 0.0
    runup = _clip01(((drift or 0.0) - DIST_RUNUP_LO) / (DIST_RUNUP_HI - DIST_RUNUP_LO))
    volspike = _clip01(((vol_ratio or 0.0) - DIST_VOL_LO) / (DIST_VOL_HI - DIST_VOL_LO))
    return 100.0 * high * (0.5 + 0.3 * runup + 0.2 * volspike)


def _distribution_model_features(
    technical_distribution_score: Optional[float],
    pattern_codes: Sequence[str],
    lhb_recent: bool = False,
    divergence_score: Optional[float] = None,
) -> Dict[str, float]:
    codes = set(pattern_codes or [])
    return {
        "p14": 1.0 if "P14" in codes else 0.0,
        "p15": 1.0 if "P15" in codes else 0.0,
        "p16": 1.0 if "P16" in codes else 0.0,
        "p17": 1.0 if "P17" in codes else 0.0,
        "p19": 1.0 if "P19" in codes else 0.0,
        "p20": 1.0 if "P20" in codes else 0.0,
        "p22": 1.0 if "P22" in codes else 0.0,
        "p26": 1.0 if "P26" in codes else 0.0,
        "lhb_recent": 1.0 if lhb_recent else 0.0,
        "technical": _clip01((technical_distribution_score or 0.0) / 100.0),
        "divergence": _clip01((divergence_score or 0.0) / 100.0),
    }


def _distribution_model_score(features: Dict[str, float],
                              weights: Optional[Dict[str, float]] = None) -> float:
    weights = weights or DIST_MODEL_WEIGHTS
    return round(100.0 * sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in weights), 1)


def _pool_model_weights(base: Dict[str, float], pool: str,
                        excluded_for_etf: Sequence[str]) -> Dict[str, float]:
    """ETF 剔除公司行为因子后，把剩余技术权重重新归一到 1。"""
    if pool != "etf":
        return dict(base)
    excluded = set(excluded_for_etf)
    total = sum(float(weight) for key, weight in base.items() if key not in excluded)
    if total <= 0:
        return {}
    return {
        key: float(weight) / total
        for key, weight in base.items()
        if key not in excluded
    }


def accumulation_model_weights(pool: str = DEFAULT_POOL) -> Dict[str, float]:
    return _pool_model_weights(ACCUM_MODEL_WEIGHTS, pool, ETF_ACCUM_EXCLUDED)


def distribution_model_weights(pool: str = DEFAULT_POOL) -> Dict[str, float]:
    return _pool_model_weights(DIST_MODEL_WEIGHTS, pool, ETF_DIST_EXCLUDED)


def _apply_distribution_model(row: Dict[str, Any], pool: str = DEFAULT_POOL) -> None:
    signals = row.setdefault("signals", {})
    pattern_codes = row.get("patterns") or []
    raw_features = _distribution_model_features(
        signals.get("technical_distribution_score", row.get("distribution_score")),
        pattern_codes,
        bool(row.get("lhb_recent")),
        (row.get("sub_scores") or {}).get("divergence"),
    )
    weights = distribution_model_weights(pool)
    features = {key: raw_features[key] for key in weights}
    score = _distribution_model_score(features, weights)
    row["distribution_score"] = score
    signals["distribution_score"] = score
    signals["distribution_model_features"] = {k: round(v, 4) for k, v in features.items()}
    signals["distribution_model_weights"] = {k: round(float(v), 4) for k, v in weights.items()}
    if pool == "etf":
        signals["distribution_model_excluded"] = list(ETF_DIST_EXCLUDED)


def _score_bars(
    code: str,
    bars: List[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """提取雷达需要的技术原始分与形态上下文；数据不足返回 None。

    这里继续计算 position/chip/cmf_eff 等原始特征；chip 同时供吸筹总分、展示与形态匹配。
    不再生成旧四技术因子加权分。
    最终吸筹分统一由 ``_accumulation_model_score`` 计算。
    """
    if len(bars) < MIN_BARS:
        return None
    ctx = ctx or _build_pattern_context(code, bars)
    vol_measure = ctx.get("vol_measure")
    s_pos, close_pctile = ctx.get("score_position"), ctx.get("pos")
    s_div, drift = ctx.get("score_divergence"), ctx.get("drift")
    s_cmf, cmf = ctx.get("score_cmf"), ctx.get("cmf")
    s_chip, chip = ctx.get("score_chip"), ctx.get("chip")
    vol_ratio = ctx.get("vol_ratio")
    sealed, streak = int(ctx.get("sealed") or 0), int(ctx.get("streak") or 0)
    triggered, trigger = bool(ctx.get("triggered")), ctx.get("trigger")

    # CMF 在十因子模型中反向计入：高买压是反转风险，低/中性买压最多给50分。
    cmf_eff = min(50.0, 100.0 - s_cmf) if s_cmf is not None else None
    technical_dist = _score_distribution(close_pctile, drift, vol_ratio)
    turnover_pctile = _turnover_pctile(bars)              # 最新换手率分位（拥挤度，仅展示）
    return {
        "distribution_score": round(technical_dist, 1),
        "sealed": sealed,
        "streak": streak,
        "triggered": triggered,
        "signals": {
            "vol_measure": vol_measure,
            "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "close_pctile": round(close_pctile, 2) if close_pctile is not None else None,
            "drift_20d": round(drift, 3) if drift is not None else None,
            "cmf": round(cmf, 3) if cmf is not None else None,
            "chip_concentration": round(chip["concentration"], 2) if chip else None,
            "chip_winner": round(chip["winner"], 2) if chip else None,
            "chip_peak_price": round(chip["peak_price"], 2) if chip else None,
            "chip_peak_pctile": round(chip["peak_pctile"], 2) if chip else None,
            "chip_price_to_peak": round(chip["price_to_peak"], 3) if chip and chip["price_to_peak"] is not None else None,
            "technical_distribution_score": round(technical_dist, 1),
            "distribution_score": round(technical_dist, 1),
            "triggered": triggered,
            "trigger": trigger,
            "sealed_recent": sealed,
            "limit_streak": streak,
            "latest_turnover": bars[-1]["turnover"],
            "turnover_pctile": round(turnover_pctile, 2) if turnover_pctile is not None else None,
        },
        "sub_scores": {
            "position": round(s_pos, 1) if s_pos is not None else None,
            "divergence": round(s_div, 1) if s_div is not None else None,
            "cmf": round(s_cmf, 1) if s_cmf is not None else None,
            "cmf_eff": round(cmf_eff, 1) if cmf_eff is not None else None,
            "chip": round(s_chip, 1) if s_chip is not None else None,
            "chip_concentration": round(chip["sub_concentration"], 1) if chip else None,
            "chip_peak_low": round(chip["sub_peak_low"], 1) if chip else None,
            "chip_price_near_peak": round(chip["sub_price_near_peak"], 1) if chip else None,
            "chip_winner_mid_low": round(chip["sub_winner_mid_low"], 1) if chip else None,
        },
    }


def _state_label(score: Optional[float], sealed: int, streak: int,
                 triggered: bool = False, phase: Optional[str] = None) -> str:
    """兼容旧 state 字段；疑似吸筹以主阶段判定为准，纯分数档位不复用同名标签。"""
    if sealed > 0 or streak >= 4:
        return "已启动(封板/连板,非吸筹)"
    if score is None:
        return "数据不足"
    if triggered and score >= 40:
        return "放量突破(右进)"      # 事件标记；verify 显示截面不占优，慎追
    if phase and "疑似吸筹" in phase:
        return "疑似吸筹(待确认)"
    if score >= 65:
        return "吸筹特征较强"
    if score >= 40:
        return "吸筹特征中等"
    return "平淡"


def _clip_score(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _phase_pattern_strength(fired: List[Dict[str, str]], signals: set) -> float:
    """当前阶段相关形态强度：有效形态权重大，其余形态仍给有限确认。"""
    support = 0.0
    for p in fired:
        if p.get("signal") not in signals:
            continue
        support += 1.25 if p.get("code") in PATTERN_EFFECTIVE else 0.85
    if support <= 0.0:
        return 0.0
    return _clip_score(35.0 + 18.0 * support)


def _phase_confidence(phase: str, fired: List[Dict[str, str]],
                      ambush_score: float, distribution_score: float) -> float:
    """阶段把握度（连续 0~100）。

    阶段标签由形态负责；把握度衡量“这个阶段判断是否被分数确认”：
      - 吸筹/洗盘：看买入形态 + 吸筹分，出货分高则扣分；
      - 出货：形态负责触发；把握用PIT样本外校准后的吸筹分/连续出货分Logistic概率；
      - 突破/拉升/试盘：看 hold 形态 + 低出货风险，避免把追高形态机械给高分；
      - 疑似吸筹/观望：无形态时分别按“弱吸筹确认”和“无明显阶段信号”计算。
    """
    ambush = _clip_score(ambush_score or 0.0)
    dist = _clip_score(distribution_score or 0.0)
    safe = 100.0 - dist

    if "疑似吸筹" in phase:
        return round(min(75.0, 0.70 * ambush + 0.30 * safe), 1)
    if "观望" in phase or "空仓" in phase:
        signal_strength = max(ambush, dist)
        return round(_clip_score(100.0 - 1.15 * signal_strength, 0.0, 85.0), 1)

    if "出货" in phase:
        logit = (
            PHASE_CONFIDENCE_SELL_INTERCEPT
            + PHASE_CONFIDENCE_SELL_ACCUMULATION_COEF * ambush / 100.0
            + PHASE_CONFIDENCE_SELL_DISTRIBUTION_COEF * dist / 100.0
        )
        # 数值稳定的 sigmoid；正常0~100输入对应的logit远未到溢出区，但保留保护。
        if logit >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-min(logit, 60.0)))
        else:
            exp_logit = math.exp(max(logit, -60.0))
            probability = exp_logit / (1.0 + exp_logit)
        return round(100.0 * probability, 1)

    if "突破" in phase or "拉升" in phase or "试盘" in phase:
        shape = _phase_pattern_strength(fired, {"hold"})
        risk_penalty = max(0.0, dist - 30.0) * 0.45
        return round(_clip_score(0.50 * shape + 0.25 * ambush + 0.25 * safe - risk_penalty), 1)

    # 吸筹 / 洗盘 / 吸筹+洗盘：形态与吸筹分同向才给高把握，出货分越高越降权。
    shape = _phase_pattern_strength(fired, {"buy"})
    conflict_penalty = max(0.0, dist - 35.0) * 0.40
    return round(_clip_score(0.45 * shape + 0.45 * ambush + 0.10 * safe - conflict_penalty), 1)


def _phase_invalidations(phase: str) -> List[str]:
    """每个阶段的证伪/止损条件（沿用早期形态研究的 invalidations，让标签可执行）。"""
    if "出货" in phase:
        return ["缩量回踩后重新放量突破前高才解除风险", "跌破高位平台下沿=派发确认"]
    if "突破" in phase:
        return ["次日低开低走或突破位站不稳3日", "突破后回到平台内=假突破"]
    if "拉升" in phase:
        return ["跌破5/10日线且放量", "高位放量滞涨或长上影=见顶"]
    if "试盘" in phase:
        return ["试盘后无量回落、不能再创异动高点", "跌破试盘当日低点"]
    if "吸筹" in phase or "洗盘" in phase:                    # 含疑似吸筹/吸筹+洗盘
        return ["跌破筑底箱体下沿且3-5日不收回", "放量下跌且筹码峰发散", "所属题材退潮"]
    return ["无形态信号：仅观察，待左侧吸筹形态出现再评估"]   # 观望


def _evidence(res: Dict[str, Any], fired: List[Dict[str, str]],
              bars: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    """把吸筹分的子分/信号翻译成结构化「依据」标签：[{label, kind}]，kind ∈ bullish/neutral/bearish。

    只verbalize 因子层（位置/背离/CMF/筹码/压缩/换手/派发）——形态层由 P 编号 + 解释模块呈现，互补不重复。
    前端按 kind 上色（利好绿/中性灰/风险红），替代原先靠关键字正则猜色。
    """
    sub = res.get("sub_scores", {})
    sig = res.get("signals", {})
    tags: List[Dict[str, str]] = []
    def add(label: str, kind: str) -> None:
        tags.append({"label": label, "kind": kind})

    if (sub.get("position") or 0) >= 60:
        add("价格中低位", "bullish")
    # 注：背离(放量价稳)/CMF(买压) 已在吸筹分里「反向计入」(实测负 edge)，不再作为利好 evidence；
    #     高背离/高CMF 反而是轻度反转风险，故标 bearish。
    if (sub.get("divergence") or 0) >= 60:
        add("放量滞涨(反转风险)", "bearish")
    if (sub.get("cmf") or 0) >= 60:
        add("高位买压(反转风险)", "bearish")
    if (sub.get("chip") or 0) >= 60:
        add("低位筹码集中", "bullish")
    fired_codes = {pattern.get("code") for pattern in fired}
    pos = sig.get("close_pctile")
    if "P23" in fired_codes:
        add("低位压缩温和量能转强", "bullish")
    elif bars is not None and pos is not None and pos < P23_POSITION_60_MAX:
        ratio = _amp_ratio(bars)
        if ratio is not None and ratio < COMPRESS_AMP_RATIO:
            add("波动压缩(待企稳确认)", "neutral")
    if sig.get("triggered"):
        add("放量突破(追高风险)", "bearish")       # P11 全历史逐日复测为双池负向风险
    streak = sig.get("limit_streak") or 0
    if streak >= 2:
        add(f"连板{streak}(已启动)", "neutral")
    elif sig.get("sealed_recent"):
        add("一字封板(买不进)", "neutral")
    tp = sig.get("turnover_pctile")
    if tp is not None and tp > 0.84:
        add("换手拥挤", "bearish")
    if "P26" in fired_codes:
        winner = sig.get("chip_winner")
        change = sig.get("chip_winner_5d_change")
        if winner is not None and winner >= CHIP_WINNER_RISK_HIGH:
            add(f"获利盘{winner:.0%}(兑现风险)", "bearish")
        elif change is not None:
            add(f"获利盘5日+{change:.0%}(快速转盈)", "bearish")
    if (sig.get("technical_distribution_score", sig.get("distribution_score")) or 0) >= 40:
        add("高位派发风险", "bearish")
    if any(p["signal"] == "sell" for p in fired) and (sig.get("technical_distribution_score", sig.get("distribution_score")) or 0) < 40:
        add("出货形态(破位/见顶)", "bearish")
    return tags


def _score_candidate_from_bars(
    cand: Dict[str, Any],
    bars: List[Dict[str, Any]],
    pool: str = DEFAULT_POOL,
    p1_market_gate: bool = False,
    p1_state_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """给单只候选按给定 bars 打分；离线/实时路径共用同一套形态逻辑。"""
    out = dict(cand)
    current_bars = bars[-LOOKBACK:]
    if len(current_bars) < MIN_BARS:
        p1_state = (
            dict(p1_state_override)
            if isinstance(p1_state_override, dict)
            else _p1_rule_state(bars, p1_market_gate if pool != "etf" else False)
        )
        out.update({"ambush_score": None, "score_status": "INSUFFICIENT_DATA",
                    "state": "数据不足", "last_date": bars[-1]["date"] if bars else None,
                    "p1_rule_state": p1_state})
        return out
    pattern_context = _build_pattern_context(cand["code"], bars)
    res = _score_bars(cand["code"], current_bars, ctx=pattern_context)
    if res is None:
        p1_state = (
            dict(p1_state_override)
            if isinstance(p1_state_override, dict)
            else _p1_rule_state(bars, p1_market_gate if pool != "etf" else False)
        )
        out.update({"ambush_score": None, "score_status": "INSUFFICIENT_DATA",
                    "state": "数据不足", "last_date": bars[-1]["date"] if bars else None,
                    "p1_rule_state": p1_state})
        return out
    p1_state = (
        dict(p1_state_override)
        if isinstance(p1_state_override, dict)
        else _p1_rule_state(bars, p1_market_gate if pool != "etf" else False)
    )
    fired = match_patterns(
        cand["code"], bars, ctx=pattern_context, pool=pool,
        p1_market_gate=p1_market_gate if pool != "etf" else False,
        p1_override=bool(p1_state.get("active")),
    )
    winner = pattern_context.get("chip_winner")
    winner_prior = pattern_context.get("chip_winner_prior")
    winner_change = pattern_context.get("chip_winner_change")
    effective_p1_market_gate = bool(
        p1_state.get("market_gate")
        if isinstance(p1_state_override, dict)
        else (p1_market_gate if pool != "etf" else False)
    )
    res["signals"].update({
        "p1_market_gate": effective_p1_market_gate,
        "p1_signal_date": p1_state.get("signal_date"),
        "p1_source": p1_state.get("source"),
        "p1_base_leg_hit": bool(p1_state.get("base_leg_hit")),
        "p1_breakout_addon_hit": bool(p1_state.get("supplemental_hit")),
        "p1_eod_only": True,
        "chip_winner": round(winner, 2) if winner is not None else None,
        "chip_winner_5d_ago": round(winner_prior, 2) if winner_prior is not None else None,
        "chip_winner_5d_change": round(winner_change, 3) if winner_change is not None else None,
        "p26_volume_ratio": round(pattern_context["p26_volume_ratio"], 2)
        if pattern_context.get("p26_volume_ratio") is not None else None,
        "chip_winner_risk": bool("P26" in {pattern["code"] for pattern in fired}),
    })
    pattern_codes = [p["code"] for p in fired]
    distribution_warning_points = _distribution_warning_points(fired)
    distribution_warning_effective_sell_count = _effective_sell_pattern_count(fired)
    distribution_warning_triggered = _distribution_warning_triggered(fired)
    res["patterns"] = pattern_codes
    _apply_distribution_model(res, pool=pool)
    score_features = {
        "sub_scores": res.get("sub_scores") or {},
        "patterns": pattern_codes,
        "holder_change": out.get("holder_change"),
        "repurchase_recent": bool(out.get("repurchase_recent")),
    }
    weights = accumulation_model_weights(pool)
    raw_score_features = _accumulation_raw_features(score_features)
    active_score_features = {key: raw_score_features[key] for key in weights}
    score = _accumulation_model_score(active_score_features, weights)
    phase = _pattern_phase(fired, score)
    out.update({
        "ambush_score": score,
        "distribution_score": res["distribution_score"],
        "score_status": "OK",
        "triggered": res["triggered"],
        "state": _state_label(score, res["sealed"], res["streak"], res["triggered"], phase),
        "last_date": bars[-1]["date"],
        "patterns": pattern_codes,
        "pattern_detail": fired,
        "p1_rule_state": p1_state,
        "distribution_warning_points": distribution_warning_points,
        "distribution_warning_effective_sell_count": distribution_warning_effective_sell_count,
        "distribution_warning_triggered": distribution_warning_triggered,
        "pattern_phase": phase,
        "phase_confidence": _phase_confidence(phase, fired, score, res["distribution_score"]),
        "invalidations": _phase_invalidations(phase),
        "evidence": _evidence(res, fired, current_bars),
        "signals": res["signals"],
        "sub_scores": res["sub_scores"],
    })
    out["signals"]["distribution_warning_points"] = distribution_warning_points
    out["signals"]["distribution_warning_points_explanation_only"] = False
    out["signals"]["distribution_warning_points_threshold"] = (
        DISTRIBUTION_WARNING_POINTS_THRESHOLD
    )
    out["signals"]["distribution_warning_effective_sell_count"] = (
        distribution_warning_effective_sell_count
    )
    out["signals"]["distribution_warning_effective_sell_threshold"] = (
        DISTRIBUTION_WARNING_EFFECTIVE_SELL_THRESHOLD
    )
    out["signals"]["distribution_warning_rule"] = distribution_warning_rule_metadata()
    out["signals"]["distribution_warning_triggered"] = distribution_warning_triggered
    out["signals"]["factor_applicability"] = {
        "company_capital": pool != "etf",
        "chip": out["signals"].get("chip_concentration") is not None,
        "technical": True,
    }
    if pool == "etf":
        out["signals"]["accumulation_model_excluded"] = list(ETF_ACCUM_EXCLUDED)
        out["signals"]["distribution_model_excluded"] = list(ETF_DIST_EXCLUDED)
    out.update({f"rev_{k}": v for k, v in _reversal_raw_features(cand["code"], current_bars).items()})
    if pool == "hotmoney" and REVERSAL_SMOOTH_DAYS > 1:
        # 近 N 日各自的原始过热因子(0=今日)；PIT：第 k 日窗口=砍掉最后 k 根 bar，仍 ≤ as_of。
        hist: List[Dict[str, Any]] = []
        for k in range(REVERSAL_SMOOTH_DAYS):
            sub = current_bars if k == 0 else current_bars[:-k]
            if len(sub) < MIN_BARS:
                break
            hist.append(_reversal_raw_features(cand["code"], sub))
        out["rev_hist"] = hist
    return out


def score_candidate(
    conn: sqlite3.Connection,
    cand: Dict[str, Any],
    as_of: Optional[str] = None,
    pool: str = DEFAULT_POOL,
    p1_market_gate: Optional[bool] = None,
) -> Dict[str, Any]:
    """给单只龙头算当下潜伏分 + 游资形态匹配，返回带子分/原始信号/形态的明细行。

    as_of 给定时只用该日期及以前的 bar（PIT 防泄漏，供历史复盘）。
    pool='hotmoney' 且开启平滑(REVERSAL_SMOOTH_DAYS>1)时附 rev_hist(近 N 日原始过热因子,供 EMA 平滑)。
    """
    bars = _recent_bars(conn, cand["code"], limit=PATTERN_EVAL_BARS, as_of=as_of)
    if p1_market_gate is None:
        signal_date = str(bars[-1].get("date") or "") if bars else ""
        p1_market_gate = _p1_market_gate_by_date(conn, [signal_date]).get(signal_date, False)
    return _score_candidate_from_bars(
        cand,
        bars,
        pool=pool,
        p1_market_gate=bool(p1_market_gate),
    )


# ── 输出 ──────────────────────────────────────────────────────

def base_payload(mode: str, candidate_count: int,
                 pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    sources = {
        "leader": "sw3_member.is_leader (细分行业龙头池)",
        "hotmoney": "sw3_member.is_hot_money (游资小盘池)",
        "etf": "stock_etf_pool.py fund_index (ETF配置池)",
    }
    return {
        "schema": SCHEMA,
        "mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": sources.get(pool, sources[DEFAULT_POOL]),
        "candidate_count": candidate_count,
    }


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


# ── ambush ────────────────────────────────────────────────────

THEME_STALE_DAYS = 7        # 题材数据超过该天数视为偏旧（热度时效性强）


def _clean_theme_name(value: Any) -> str:
    return str(value or "").strip()


def _load_theme_map() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    """读取 stock_theme_candidates.py 落盘的题材映射。

    返回 (映射, 元信息)：
      映射 = {code: {theme: 拟合二级行业名, theme_code, heat_pctile: 行业热度百分位}}；
      真实二级行业热度 = {plate_name: heat_pctile}；
      元信息 = {available, generated_at, age_days, stale}。
    热度百分位由题材热度排名换算（rank 越靠前越接近 100，rank=1→100、rank=N→0）。
    文件缺失/解析失败则映射为空、available=False，雷达照常出表（热度列显示空值）。
    """
    meta: Dict[str, Any] = {"available": False, "generated_at": None, "age_days": None, "stale": False}
    try:
        data = json.loads(THEME_CANDIDATES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, meta, {}
    meta["available"] = True
    gen = data.get("generated_at")
    meta["generated_at"] = gen
    if gen:
        try:
            age = (datetime.now() - datetime.strptime(gen, "%Y-%m-%d %H:%M:%S")).days
            meta["age_days"] = age
            meta["stale"] = age >= THEME_STALE_DAYS
        except ValueError:
            pass
    rankings = data.get("theme_rankings") or []
    n = len(rankings)
    pctile_by_code: Dict[str, float] = {}
    pctile_by_name: Dict[str, float] = {}
    for row in rankings:
        rank, plate_code = row.get("rank"), row.get("plate_code")
        if rank is None:
            continue
        pctile = 100.0 if n <= 1 else (n - rank) / (n - 1) * 100.0
        if plate_code is not None:
            pctile_by_code[str(plate_code)] = pctile
        plate_name = _clean_theme_name(row.get("plate_name"))
        if plate_name:
            pctile_by_name[plate_name] = pctile
    out: Dict[str, Dict[str, Any]] = {}
    for st in data.get("stock_themes") or []:
        code = stock_storage._normalize_code(st.get("code"))
        if not code:
            continue
        theme_code = st.get("tracking_theme_code")
        out[code] = {
            "theme": st.get("tracking_theme") or "",
            "theme_code": theme_code,
            "heat_pctile": pctile_by_code.get(str(theme_code)),
        }
    return out, meta, pctile_by_name


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _latest_stock_history_date(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(date) AS latest_date FROM stock_history WHERE daily_close IS NOT NULL"
    ).fetchone()
    return str(row["latest_date"]) if row and row["latest_date"] else None


def _load_capital_map(conn: sqlite3.Connection, as_of: Optional[str]) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """资金面信号 PIT 聚合 → {code: {holder_change, repurchase_recent, lhb_recent}}。

    PIT：as_of(默认最新交易日)当日及以前可见的信息。
      - holder_change：as_of 前最近一期已公告的股东户数增减比例(%)，负=户数降=看多。
      - repurchase_recent：as_of 前 CAPITAL_EVENT_DAYS 日内有回购公告(弱买点)。
      - lhb_recent：as_of 前 CAPITAL_EVENT_DAYS 日内上过龙虎榜(反向避雷)。
    数据层缺失(表不存在)则返回 ({}, False)，雷达照常出表（户数分按中性、回购分按0处理）。
    """
    have = _table_exists(conn, HOLDER_TABLE) or _table_exists(conn, REPURCHASE_TABLE) or _table_exists(conn, LHB_TABLE)
    if not have:
        return {}, False
    asof = as_of or _latest_stock_history_date(conn) or datetime.now().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=CAPITAL_EVENT_DAYS)).strftime("%Y-%m-%d")
    out: Dict[str, Dict[str, Any]] = {}

    def slot(code: str) -> Dict[str, Any]:
        return out.setdefault(code, {"holder_change": None, "repurchase_recent": False, "lhb_recent": False})

    if _table_exists(conn, HOLDER_TABLE):
        for row in conn.execute(
            f"SELECT code, change_pct FROM {HOLDER_TABLE} "
            "WHERE disclose_date IS NOT NULL AND change_pct IS NOT NULL AND disclose_date <= ? "
            "ORDER BY code, disclose_date", (asof,)
        ).fetchall():
            slot(stock_storage._normalize_code(row["code"]))["holder_change"] = row["change_pct"]  # 升序→最后一条最新
    if _table_exists(conn, REPURCHASE_TABLE):
        for row in conn.execute(
            f"SELECT DISTINCT code FROM {REPURCHASE_TABLE} WHERE disclose_date > ? AND disclose_date <= ?",
            (cutoff, asof)
        ).fetchall():
            slot(stock_storage._normalize_code(row["code"]))["repurchase_recent"] = True
    if _table_exists(conn, LHB_TABLE):
        for row in conn.execute(
            f"SELECT DISTINCT code FROM {LHB_TABLE} WHERE date > ? AND date <= ?", (cutoff, asof)
        ).fetchall():
            slot(stock_storage._normalize_code(row["code"]))["lhb_recent"] = True
    return out, True


def _holder_change_score(holder_change: Optional[float]) -> float:
    """股东户数变化原始分：-15%(户数大降)→100、+15%(户数大增)→0；缺失按中性50。"""
    if holder_change is None:
        return 50.0
    return _clip01((15.0 - float(holder_change)) / 30.0) * 100.0


def _repurchase_score(repurchase_recent: bool) -> float:
    """公司回购原始分：近 CAPITAL_EVENT_DAYS 日有回购公告给满分，否则为 0。"""
    return 100.0 if repurchase_recent else 0.0


def _accumulation_raw_features(row: Dict[str, Any]) -> Dict[str, float]:
    sub = row.get("sub_scores") or {}
    pattern_codes = set(row.get("patterns") or [])
    return {
        "chip": float(sub.get("chip") or 0.0),
        "position": float(sub.get("position") or 0.0),
        "cmf_eff": float(sub.get("cmf_eff") or 0.0),
        "p2": 100.0 if "P2" in pattern_codes else 0.0,
        "p3": 100.0 if "P3" in pattern_codes else 0.0,
        "p5": 100.0 if "P5" in pattern_codes else 0.0,
        "p21": 100.0 if "P21" in pattern_codes else 0.0,
        "p23": 100.0 if "P23" in pattern_codes else 0.0,
        "p24": 100.0 if "P24" in pattern_codes else 0.0,
        "p1": 100.0 if "P1" in pattern_codes else 0.0,
        "p25": 100.0 if "P25" in pattern_codes else 0.0,
        "holder_change": _holder_change_score(row.get("holder_change")),
        "repurchase": _repurchase_score(bool(row.get("repurchase_recent"))),
    }


def _accumulation_model_score(features: Dict[str, float],
                              weights: Optional[Dict[str, float]] = None) -> float:
    weights = weights or ACCUM_MODEL_WEIGHTS
    return round(sum(float(weights.get(k, 0.0)) * float(features.get(k, 0.0)) for k in ACCUM_FEATURES), 1)


def _percentiles_from_values(values: Sequence[Any]) -> Dict[int, float]:
    """一列值 → {原始索引: 0-100 百分位}；同分共享平均名次，None/非数跳过(不计百分位)。"""
    pairs: List[Tuple[int, float]] = []
    for idx, value in enumerate(values):
        s = _safe(value)
        if s is not None:
            pairs.append((idx, s))
    if not pairs:
        return {}
    if len(pairs) == 1:
        return {pairs[0][0]: 50.0}
    pairs.sort(key=lambda item: item[1])
    n = len(pairs)
    percentiles: Dict[int, float] = {}
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and pairs[end][1] == pairs[pos][1]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0
        percentile = round(avg_rank / (n - 1) * 100.0, 1)
        for i in range(pos, end):
            percentiles[pairs[i][0]] = percentile
        pos = end
    return percentiles


def _cross_section_percentiles(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    """按当前候选横截面给分数转 0-100 百分位；同分共享平均名次。"""
    return _percentiles_from_values([row.get(key) for row in rows])


def _ema_recent_first(seq: Sequence[float], span: int) -> float:
    """EMA(seq[0]=最新)。span 越大越平滑、今日权重最高；单点/seq 长度 1 时即原值。"""
    if not seq:
        return 0.0
    alpha = 2.0 / (span + 1)
    acc = seq[-1]                       # 最旧
    for v in reversed(seq[:-1]):        # 由旧到新，最后并入今日
        acc = alpha * v + (1.0 - alpha) * acc
    return acc


def _opportunity_score(accumulation_percentile: Optional[float],
                       distribution_percentile: Optional[float]) -> Optional[float]:
    """机会分：吸筹/出货先转截面百分位，再用出货风险折扣吸筹强度。"""
    accumulation = _safe(accumulation_percentile)
    if accumulation is None:
        return None
    dist = _clip_score(_safe(distribution_percentile) or 0.0)
    penalty = _clip01(OPPORTUNITY_DISTRIBUTION_PENALTY)
    return round(_clip_score(accumulation) * (1.0 - penalty * dist / 100.0), 1)


def _apply_accumulation_model(rows: List[Dict[str, Any]],
                              pool: str = DEFAULT_POOL) -> None:
    """吸筹/出货保留原始解释分；机会分使用横截面百分位合成。"""
    weights = accumulation_model_weights(pool)
    for r in rows:
        signals = r.setdefault("signals", {})
        raw_features = _accumulation_raw_features(r)
        features = {key: raw_features[key] for key in weights}
        score = _accumulation_model_score(features, weights)
        r["ambush_score"] = score
        signals["accumulation_model_features"] = {k: round(v, 1) for k, v in features.items()}
        signals["accumulation_model_weights"] = {k: round(float(v), 4) for k, v in weights.items()}
        if pool == "etf":
            signals["accumulation_model_excluded"] = list(ETF_ACCUM_EXCLUDED)
            signals.pop("holder_change_score", None)
            signals.pop("repurchase_score", None)
        else:
            signals["holder_change_score"] = round(features["holder_change"], 1)
            signals["repurchase_score"] = round(features["repurchase"], 1)

    accumulation_percentiles = _cross_section_percentiles(rows, "ambush_score")
    distribution_percentiles = _cross_section_percentiles(rows, "distribution_score")
    for idx, r in enumerate(rows):
        score = r.get("ambush_score")
        signals = r.setdefault("signals", {})
        accumulation_pct = accumulation_percentiles.get(idx)
        distribution_pct = distribution_percentiles.get(idx)
        opportunity = _opportunity_score(accumulation_pct, distribution_pct)
        r["accumulation_percentile"] = accumulation_pct
        r["distribution_percentile"] = distribution_pct
        r["opportunity_score"] = opportunity
        signals["accumulation_percentile"] = accumulation_pct
        signals["distribution_percentile"] = distribution_pct
        signals["opportunity_score"] = opportunity
        signals["opportunity_formula"] = OPPORTUNITY_FORMULA
        fired = r.get("pattern_detail") or []
        phase = _pattern_phase(fired, score)
        r["pattern_phase"] = phase
        r["phase_confidence"] = _phase_confidence(phase, fired, score, r.get("distribution_score") or 0.0)
        r["invalidations"] = _phase_invalidations(phase)
        r["state"] = _state_label(
            score,
            int(signals.get("sealed_recent") or 0),
            int(signals.get("limit_streak") or 0),
            bool(r.get("triggered")),
            phase,
        )


def _attach_capital_evidence(row: Dict[str, Any]) -> None:
    ev = row.get("evidence")
    if not isinstance(ev, list):
        return
    hc = row.get("holder_change")
    if hc is not None and hc <= -5:
        ev.append({"label": f"股东户数降{abs(hc):.0f}%", "kind": "bullish"})
    if row.get("repurchase_recent"):
        ev.append({"label": "公司回购中", "kind": "bullish"})
    if row.get("lhb_recent"):
        ev.append({"label": "近期上龙虎榜，波动加剧", "kind": "bearish"})


# ── 游资池超短反转分（hotmoney 专用，纪要14）──────────────────
# 游资小盘池 3 日尺度无可交易动量、反转极强(verify: 反转分 IC3d +0.101/t7.4、多空 +0.68%/3日，
#   首个多空转正信号)。分 = 把"过热"因子(换手分位/振幅/涨停/量比/动量)截面 rank 后【反向】加权——
#   买不拥挤/不放量/不大振幅/没涨停/没追涨的票，捕捉超短反弹。
#   ⚠ top 档纯做多 3日胜率仅 48%(<50%，回测期偏熊 base rate)，宜多空对冲或叠加市场状态择时。
REVERSAL_WEIGHTS = {
    "turn_pctile": 0.30,   # 换手分位(最强反转, IC3d t-7.5)
    "amp_today": 0.20,     # 当日振幅(t-6.4)
    "limitup_5d": 0.20,    # 近5日涨停数(t-6.8)
    "vol_ratio": 0.15,     # 量比(t-4.2)
    "mom_5d": 0.15,        # 近5日动量(t-2.8)
}
REVERSAL_FEATURES = tuple(REVERSAL_WEIGHTS)
# 反转分平滑(纪要15)：对近 REVERSAL_SMOOTH_DAYS 日各算一次截面反向 rank 子分，再 EMA(span=该值)合成，
#   今日权重最高。1 = 关闭平滑/单日快照。消融(2026-06 重建 hotmoney 池 543 只)：
#   · 全样本：单日 ≈ ema3(IC 打平 3日~0.11)，平滑仅把多空价差抬一丢丢；
#   · favorable(大盘>MA20,真做多场) + 近段(OOS 后半)平滑占优：3日 IC 0.107→0.121(t7.1→8.7)、多空 +0.44%→+0.68%；
#   · 前半强 regime 单日略优(平滑稀释最新极值)。ema3 = 稳健补丁(二阶,前半几乎不掉、近段/favorable 吃增益)。
REVERSAL_SMOOTH_DAYS = 3

# 潜伏妖股筛选（latent 模式）：
#   从"太极 2022 式"左侧指纹反推——低位 + 安静(没被发现) + 吸筹指纹 + 妖股基因 + 真吸筹证据。
#   ⚠️ 左侧买入侧本就弱/慢(IC~0.04, regime 依赖, 纪要12)→ 这是观察名单，不是买点触发器。
LATENT_MAX_POS = 0.35           # 收盘价近60日分位上限(左侧低位)
LATENT_MAX_TURN_PCTILE = 0.60   # 最新换手率窗口分位上限(安静、不拥挤)
LATENT_MAX_DIST = 30.0          # 出货分上限(排除已 topping)
LATENT_GENE_LOOKBACK_YEARS = 3  # 妖股基因:近 N 年最大换手/涨停数统计窗口
LATENT_WEIGHTS = {              # 综合潜伏分权重(吸筹状态为主)
    "accumulation": 0.40,       # 吸筹分截面分位(0~1)
    "gene": 0.25,               # 妖股基因(全史龙虎榜次数+近3年最大换手+涨停数, 池内分位等权)
    "low_pos": 0.15,            # 越低位越好 = 1 − 位置
    "theme_heat": 0.10,         # 所属 SW2 板块题材热度(题材①的弱代理)
}
LATENT_HOLDER_BONUS = 0.10      # 最新股东户数下降(真吸筹证据)
LATENT_REPO_BONUS = 0.10        # 近90日回购(干净的公司行为吸筹)


def _reversal_raw_features(code: str, bars: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """游资反转分的原始"过热"因子（值越大=越过热=反转分越低）。"""
    vol, _ = _volume_series(bars)
    closes = [b["close"] for b in bars]
    _, vol_ratio = _score_volume_ratio(vol)
    prev_close = closes[-2] if len(closes) > 1 else None
    h, l = bars[-1]["high"], bars[-1]["low"]
    amp_today = (h - l) / prev_close if prev_close and h is not None and l is not None else None
    mom_5d = (closes[-1] / closes[-6] - 1.0) if len(closes) > 5 and closes[-1] and closes[-6] else None
    limit = _limit_pct(code) - 0.3
    limitup_5d = float(sum(1 for b in bars[-5:] if b["chg"] is not None and b["chg"] >= limit))
    return {
        "turn_pctile": _turnover_pctile(bars),
        "amp_today": amp_today,
        "limitup_5d": limitup_5d,
        "vol_ratio": vol_ratio,
        "mom_5d": mom_5d,
    }


def _reversal_daily_subscores(rows: List[Dict[str, Any]], offset: int) -> Dict[int, float]:
    """近端第 offset 日(0=今日)的截面反转子分(0~100，过热因子反向 rank 加权)。

    取每行 rev_hist[offset] 的原始过热因子(缺 rev_hist 时 offset=0 退回 rev_<feat>=单日快照)；
    仅该日有原始因子的行参与当日截面排序。返回 {行索引: 子分}。
    """
    active: List[int] = []
    feats_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, r in enumerate(rows):
        hist = r.get("rev_hist")
        if hist is not None:
            feats = hist[offset] if offset < len(hist) else None
        elif offset == 0:
            feats = {f: r.get(f"rev_{f}") for f in REVERSAL_FEATURES}
        else:
            feats = None
        if feats is not None:
            active.append(idx)
            feats_by_idx[idx] = feats
    acc = {idx: 0.0 for idx in active}
    for feat, w in REVERSAL_WEIGHTS.items():
        pcts = _percentiles_from_values([feats_by_idx[idx].get(feat) for idx in active])  # 高=过热
        for pos, idx in enumerate(active):
            acc[idx] += w * (100.0 - pcts.get(pos, 50.0))   # 反向：过热→低分
    return acc


def _apply_reversal_model(rows: List[Dict[str, Any]]) -> None:
    """游资反转分：过热因子截面 rank 反向加权(0~100)写 reversal_score。

    rows 带 rev_hist(近 REVERSAL_SMOOTH_DAYS 日各自的原始因子, 0=今日)时，对每日各算一次截面子分，
    再 EMA(span=REVERSAL_SMOOTH_DAYS)平滑(纪要15)；仅带单日 rev_<feat> 时退回快照(与旧版等价)。
    """
    days = max(1, REVERSAL_SMOOTH_DAYS)
    daily: List[List[float]] = [[] for _ in rows]   # daily[idx][k]=近端第k日子分(0=今日,升序)
    for offset in range(days):
        sub = _reversal_daily_subscores(rows, offset)
        for idx, val in sub.items():
            daily[idx].append(val)
    for idx, r in enumerate(rows):
        r.pop("rev_hist", None)
        seq = daily[idx]
        snapshot = round(seq[0], 1) if seq else None
        score = round(_ema_recent_first(seq, days), 1) if seq else 0.0
        r["reversal_score"] = score
        sig = r.setdefault("signals", {})
        sig["reversal_score"] = score
        sig["reversal_score_snapshot"] = snapshot
        sig["reversal_smooth_days"] = days
        sig["reversal_features"] = {f: r.get(f"rev_{f}") for f in REVERSAL_FEATURES}
        sig["reversal_weights"] = dict(REVERSAL_WEIGHTS)


MARKET_REGIME_INDEX = "510310"   # 沪深300 ETF(index_nav)；优先用累计净值避免分红/折算跳变


def _market_regime_series(
    conn: sqlite3.Connection, as_of: Optional[str] = None
) -> Tuple[List[Tuple[str, float, str]], int]:
    """读取PIT市场序列；每个点为(date, value, nav字段)，并返回nav回退次数。"""
    entry = stock_storage.load_index_nav(conn, MARKET_REGIME_INDEX)
    recs = (entry.get("records") if isinstance(entry, dict) else None) or []
    series: List[Tuple[str, float, str]] = []
    fallback_count = 0
    for r in recs:
        d = r.get("date")
        nav_acc = _safe(r.get("nav_acc"))
        nav = _safe(r.get("nav"))
        value = nav_acc if nav_acc is not None and nav_acc > 0 else None
        field = "nav_acc"
        if value is None and nav is not None and nav > 0:
            value = nav
            field = "nav"
        if d and value is not None and (as_of is None or d <= as_of):
            if field == "nav":
                fallback_count += 1
            series.append((d, value, field))
    series.sort()
    return series, fallback_count


def _p1_market_gate_metrics(values: Sequence[float]) -> Dict[str, Any]:
    """P1 突破补充腿的五条件 510310 风险门（包含边界、缺失即关闭）。"""
    clean = [_safe(value) for value in values]
    if (
        len(clean) < 60
        or any(value is None or value <= 0 for value in clean[-60:])
    ):
        return {
            "available": False,
            "gate": False,
            "latest": None,
            "close_over_ma20": None,
            "close_over_ma60": None,
            "ret5": None,
            "ret20": None,
            "ma20_slope5": None,
            "ma20": None,
            "ma20_5d_ago": None,
            "ma60": None,
            "above_ma60": False,
            "ma20_rising_5d": False,
            "reason": "insufficient_history",
            "failed_checks": ["history"],
        }
    latest = float(clean[-1])
    ma20 = _mean([float(value) for value in clean[-20:]])
    ma20_5d_ago = _mean([float(value) for value in clean[-25:-5]])
    ma60 = _mean([float(value) for value in clean[-60:]])
    close_over_ma20 = latest / ma20 if ma20 and ma20 > 0 else None
    close_over_ma60 = latest / ma60 if ma60 and ma60 > 0 else None
    ret5 = latest / float(clean[-6]) - 1.0
    ret20 = latest / float(clean[-21]) - 1.0
    ma20_slope5 = ma20 / ma20_5d_ago - 1.0 if ma20 and ma20_5d_ago else None
    above_ma60 = latest > ma60
    ma20_rising_5d = ma20 > ma20_5d_ago
    checks = {
        "close_over_ma20": bool(
            close_over_ma20 is not None
            and close_over_ma20 >= P1_MARKET_MA20_FLOOR
        ),
        "close_over_ma60": bool(
            close_over_ma60 is not None
            and close_over_ma60 >= P1_MARKET_MA60_FLOOR
        ),
        "ret5": P1_MARKET_RET5_MIN <= ret5 <= P1_MARKET_RET5_MAX,
        "ret20": P1_MARKET_RET20_MIN <= ret20 <= P1_MARKET_RET20_MAX,
        "ma20_slope5": bool(
            ma20_slope5 is not None
            and ma20_slope5 >= P1_MARKET_MA20_SLOPE5_MIN
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "available": True,
        "gate": not failed,
        "latest": latest,
        "ma20": ma20,
        "ma20_5d_ago": ma20_5d_ago,
        "ma60": ma60,
        "close_over_ma20": close_over_ma20,
        "close_over_ma60": close_over_ma60,
        "ret5": ret5,
        "ret20": ret20,
        "ma20_slope5": ma20_slope5,
        "above_ma60": above_ma60,
        "ma20_rising_5d": ma20_rising_5d,
        "reason": "open" if not failed else "closed",
        "failed_checks": failed,
    }


def _p1_market_gate_by_date(
    conn: sqlite3.Connection,
    dates: Sequence[str],
    require_exact_date: bool = True,
) -> Dict[str, bool]:
    """按信号日生成 PIT 市场门；缺少同日 510310 数据时默认关闭。"""
    targets = sorted(dict.fromkeys(str(date) for date in dates if date))
    if not targets:
        return {}
    series, _fallback_count = _market_regime_series(conn)
    gates: Dict[str, bool] = {}
    values: List[float] = []
    source_date: Optional[str] = None
    index = 0
    for target in targets:
        while index < len(series) and series[index][0] <= target:
            source_date = series[index][0]
            values.append(series[index][1])
            index += 1
        same_date = source_date == target
        gate = bool(_p1_market_gate_metrics(values)["gate"])
        gates[target] = bool(gate and (same_date or not require_exact_date))
    return gates


def _market_regime(conn: sqlite3.Connection, as_of: Optional[str] = None) -> Dict[str, Any]:
    """市场状态(PIT)：MA20反转状态及正式P1突破补充腿风险门。"""
    series, fallback_count = _market_regime_series(conn, as_of=as_of)
    if len(series) < 21:
        return {
            "available": False,
            "p1_trade_gate_available": False,
            "p1_trade_gate": False,
        }
    closes = [v for _, v, _ in series]
    last_date, last, last_field = series[-1]
    ma20 = sum(closes[-20:]) / 20.0
    above = last > ma20
    ret5 = (closes[-1] / closes[-6] - 1.0) if len(closes) > 5 else None
    p1 = _p1_market_gate_metrics(closes)
    return {
        "available": True, "index": MARKET_REGIME_INDEX, "date": last_date,
        "value_field": last_field, "value": round(last, 4), "ma20": round(ma20, 4),
        "fallback_count": fallback_count,
        "above_ma20": above, "ret5": round(ret5, 4) if ret5 is not None else None,
        "favorable": bool(above),
        "p1_trade_gate_available": bool(p1["available"]),
        "p1_trade_gate": bool(p1["gate"]),
        "p1_trade_gate_reason": p1["reason"],
        "p1_trade_gate_failed_checks": p1["failed_checks"],
        "p1_trade_gate_close_over_ma20": p1["close_over_ma20"],
        "p1_trade_gate_close_over_ma60": p1["close_over_ma60"],
        "p1_trade_gate_ret5": p1["ret5"],
        "p1_trade_gate_ret20": p1["ret20"],
        "p1_trade_gate_ma20_slope5": p1["ma20_slope5"],
        "ma60": round(p1["ma60"], 4) if p1["ma60"] is not None else None,
        "ma20_5d_ago": round(p1["ma20_5d_ago"], 4) if p1["ma20_5d_ago"] is not None else None,
        "above_ma60": bool(p1["above_ma60"]),
        "ma20_rising_5d": bool(p1["ma20_rising_5d"]),
        "note": ("大盘站上MA20：适合做多反转分(top档3日≈+0.83%)"
                 if above else "大盘跌破MA20：反转分做多收益骤降(≈+0.13%,接刀)，宜观望/对冲"),
        "p1_note": (
            "P1突破补充腿市场门开启"
            if p1["gate"] else "P1突破补充腿市场门关闭"
        ),
    }


def _market_weekdays_after(start_text: str, end_text: str) -> Optional[int]:
    """Count possible A-share sessions in ``(start, end]``.

    This deliberately treats an unknown weekday holiday as a possible session.
    A false rejection is safer than silently applying only the latest one-day
    return across a multi-session hole in local history.
    """
    try:
        start = datetime.strptime(start_text[:10], "%Y-%m-%d").date()
        end = datetime.strptime(end_text[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    if end <= start:
        return 0
    count = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def _realtime_history_has_gap(bars: List[Dict[str, Any]], quote: Dict[str, Any]) -> bool:
    if not bars:
        return False
    last_date = str(bars[-1].get("date") or "")[:10]
    quote_date = str(quote.get("quote_date") or "")[:10]
    possible_sessions = _market_weekdays_after(last_date, quote_date)
    return possible_sessions is None or possible_sessions > 1


def _quote_to_realtime_bar(
    bars: List[Dict[str, Any]],
    quote: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    price = _safe(quote.get("price"))
    quote_date = str(quote.get("quote_date") or "")[:10]
    if price is None or not quote_date or not bars:
        return None
    if _realtime_history_has_gap(bars, quote):
        return None
    last = bars[-1]
    # 本地 stock_history 是前复权(qfq)日线；实时接口返回不复权现价。
    # 为保证形态序列连续，实时 bar 的价格一律投影到本地 qfq 基准：
    #   - 同日替换时，用前一交易日 qfq close 作基准；
    #   - 新增今日时，用最后一根本地 qfq close 作基准；
    #   - close 优先按实时涨跌幅计算，open/high/low 按未复权盘口相对昨收比例缩放。
    # 成交量/额/换手率则保留 raw_*，盘中按已交易分钟投影成全天量后再给因子使用。
    basis = bars[-2] if quote_date == last.get("date") and len(bars) >= 2 else last
    basis_close = _safe(basis.get("close"))
    raw_pre_close = _safe(quote.get("pre_close"))
    change_pct = _safe(quote.get("change_pct"))
    if change_pct is None and raw_pre_close:
        change_pct = (price / raw_pre_close - 1.0) * 100.0
    if basis_close is not None and change_pct is not None:
        close = basis_close * (1.0 + change_pct / 100.0)
    elif basis_close is not None and raw_pre_close:
        close = basis_close * price / raw_pre_close
    else:
        close = price

    def adjusted_price(raw_value: Any, fallback: Optional[float] = None) -> Optional[float]:
        raw = _safe(raw_value)
        if raw is None:
            return fallback
        if basis_close is not None and raw_pre_close:
            return basis_close * raw / raw_pre_close
        if price and close:
            return close * raw / price
        return raw

    open_px = adjusted_price(quote.get("open"), close)
    high = adjusted_price(quote.get("high"), close)
    low = adjusted_price(quote.get("low"), close)
    high = max(v for v in (high, close, open_px) if v is not None)
    low = min(v for v in (low, close, open_px) if v is not None)
    same_day = quote_date == last.get("date")
    raw_volume = _safe(quote.get("volume"))
    raw_amount = _safe(quote.get("amount"))
    raw_turnover = _safe(quote.get("turnover"))
    volume_factor, volume_elapsed, volume_projected = _intraday_volume_projection(quote, now=now)

    def projected_metric(value: Optional[float]) -> Optional[float]:
        return value * volume_factor if value is not None else None

    volume = projected_metric(raw_volume)
    amount = projected_metric(raw_amount)
    turnover = projected_metric(raw_turnover)
    return {
        "date": quote_date,
        "open": round(open_px, 6) if open_px is not None else None,
        "high": round(high, 6),
        "low": round(low, 6),
        "close": round(close, 6),
        "volume": volume if volume is not None else (last.get("volume") if same_day else 0.0),
        "amount": amount if amount is not None else (last.get("amount") if same_day else None),
        "chg": change_pct,
        "turnover": turnover if turnover is not None else (last.get("turnover") if same_day else None),
        "raw_close": price,
        "raw_pre_close": raw_pre_close,
        "raw_volume": raw_volume,
        "raw_amount": raw_amount,
        "raw_turnover": raw_turnover,
        "volume_projection_factor": round(volume_factor, 4),
        "volume_elapsed_minutes": round(volume_elapsed, 2) if volume_elapsed is not None else None,
        "volume_projected": volume_projected,
        "volume_projection_method": "u_shape_intraday" if volume_projected else "none",
        "price_adjust": "qfq_intraday_from_change_pct",
    }


def _merge_realtime_quote_bars(
    bars: List[Dict[str, Any]],
    quote: Optional[Dict[str, Any]],
    limit: int = LOOKBACK,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not bars or not quote:
        return bars, False
    bar = _quote_to_realtime_bar(bars, quote)
    if not bar:
        return bars, False
    last_date = str(bars[-1].get("date") or "")
    quote_date = str(bar.get("date") or "")
    if quote_date < last_date:
        return bars, False
    merged = list(bars)
    if quote_date == last_date:
        merged[-1] = bar
    else:
        merged.append(bar)
    return merged[-limit:], True


def realtime_rescore_payload(
    payload: Dict[str, Any],
    quotes: Dict[str, Dict[str, Any]],
    fetched_at: Optional[str] = None,
) -> Dict[str, Any]:
    """用东财全 A 快照重算当前雷达 payload，不落盘。

    网络层只拉一次全 A，数据库层批量读取近端 K 线；每只股票只做 O(LOOKBACK) 的内存形态计算。
    """
    base = dict(payload or {})
    stocks = [dict(row) for row in (base.get("stocks") or []) if row.get("code")]
    fetched_at = fetched_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not stocks:
        base["realtime_quote"] = {
            "available": False,
            "source": "eastmoney_a_spot",
            "updated_at": fetched_at,
            "message": "当前雷达没有可刷新的股票",
        }
        return base

    pool = str(base.get("pool") or DEFAULT_POOL)
    codes = [str(row.get("code")).zfill(6) for row in stocks]
    conn = stock_storage.connect(DB_FILE)
    try:
        bars_by_code = _bulk_recent_bars(conn, codes, PATTERN_EVAL_BARS)
        market_regime = _market_regime(conn) if pool == "hotmoney" else None
        gate_dates = {
            str(bars[-1].get("date") or "")
            for bars in bars_by_code.values() if bars
        }
        gate_dates.update(
            str(quote.get("quote_date") or "")[:10]
            for quote in quotes.values() if quote.get("quote_date")
        )
        p1_market_gates = (
            {} if pool == "etf"
            else _p1_market_gate_by_date(conn, sorted(gate_dates))
        )
    finally:
        conn.close()

    quote_sources = Counter(str(quote.get("source") or "unknown") for quote in quotes.values())
    primary_source = quote_sources.most_common(1)[0][0] if quote_sources else "realtime_batch"
    rescored: List[Dict[str, Any]] = []
    matched_quotes = 0
    used_realtime = 0
    missing_history = 0
    history_gap_count = 0
    for old in stocks:
        code = str(old.get("code")).zfill(6)
        bars = bars_by_code.get(code) or []
        quote = quotes.get(code)
        if quote:
            matched_quotes += 1
        history_gap = bool(quote and _realtime_history_has_gap(bars, quote))
        if history_gap:
            history_gap_count += 1
        merged_bars, used_quote = _merge_realtime_quote_bars(
            bars, quote, PATTERN_EVAL_BARS
        )
        if not merged_bars:
            missing_history += 1
            row = dict(old)
            row["realtime_status"] = "NO_LOCAL_BARS"
            rescored.append(row)
            continue
        signal_date = str(merged_bars[-1].get("date") or "")
        p1_state_override: Optional[Dict[str, Any]] = None
        if used_quote:
            if isinstance(old.get("p1_rule_state"), dict):
                p1_state_override = dict(old["p1_rule_state"])
            else:
                # 老版本离线 payload 没有状态快照时，至少按其正式 P1 标签锁定日终结果；
                # 盘中 K 线不得新触发或撤销这个只在收盘后确认的事件。
                offline_hit = "P1" in set(old.get("patterns") or [])
                old_signals = old.get("signals") or {}
                p1_state_override = {
                    "code": "P1",
                    "name": "P1超额优先复合确认",
                    "version": P1_VERSION,
                    "replaced_on": P1_REPLACED_ON,
                    "signal_date": old.get("last_date"),
                    "active": offline_hit,
                    "hit": offline_hit,
                    "source": old_signals.get("p1_source") or "offline_snapshot",
                    "base_leg_hit": bool(old_signals.get("p1_base_leg_hit")),
                    "supplemental_hit": bool(old_signals.get("p1_breakout_addon_hit")),
                    "market_gate": bool(old_signals.get("p1_market_gate")),
                    "breakout_addon_state": None,
                    "eod_only": True,
                    "score_weight": ACCUM_MODEL_WEIGHTS["p1"],
                    "production_replacement": True,
                }
        row = _score_candidate_from_bars(
            old, merged_bars, pool=pool,
            p1_market_gate=p1_market_gates.get(signal_date, False),
            p1_state_override=p1_state_override,
        )
        if quote:
            row["realtime_price"] = quote.get("price")
            row["realtime_change_pct"] = quote.get("change_pct")
            row["realtime_quote_time"] = quote.get("quote_time")
            if pool == "etf" and quote.get("fund_scale_yi") is not None:
                row["fund_scale_yi"] = quote.get("fund_scale_yi")
            if used_quote and merged_bars:
                row["realtime_adjusted_close"] = merged_bars[-1].get("close")
                row["realtime_price_adjust"] = merged_bars[-1].get("price_adjust")
                row["realtime_raw_volume"] = merged_bars[-1].get("raw_volume")
                row["realtime_projected_volume"] = merged_bars[-1].get("volume")
                row["realtime_raw_turnover"] = merged_bars[-1].get("raw_turnover")
                row["realtime_projected_turnover"] = merged_bars[-1].get("turnover")
                row["realtime_volume_projection_factor"] = merged_bars[-1].get("volume_projection_factor")
                row["realtime_volume_elapsed_minutes"] = merged_bars[-1].get("volume_elapsed_minutes")
                row["realtime_volume_projected"] = merged_bars[-1].get("volume_projected")
                row["realtime_volume_projection_method"] = merged_bars[-1].get("volume_projection_method")
            if pool != "etf" and quote.get("market_cap_yi") is not None:
                row["market_cap_yi"] = quote.get("market_cap_yi")
        row["realtime_status"] = (
            "LOCAL_HISTORY_GAP" if history_gap else ("UPDATED" if used_quote else "LOCAL_BARS_ONLY")
        )
        if used_quote:
            used_realtime += 1
        _apply_distribution_model(row, pool=pool)
        if pool != "etf":
            _attach_capital_evidence(row)
        rescored.append(row)

    ranked = [row for row in rescored if row.get("ambush_score") is not None]
    _apply_accumulation_model(ranked, pool=pool)
    if pool == "hotmoney":
        _apply_reversal_model(ranked)
        ranked.sort(key=lambda r: (r.get("reversal_score") or 0.0), reverse=True)
    else:
        ranked.sort(key=lambda r: (r.get("opportunity_score") or 0.0, r.get("ambush_score") or 0.0), reverse=True)

    base.update({
        "stocks": ranked,
        "scored_count": len(ranked),
        "skipped_count": len(rescored) - len(ranked),
        "triggered_count": sum(1 for row in ranked if row.get("triggered")),
        "phase_counts": dict(Counter(row.get("pattern_phase") for row in ranked)),
        "realtime_quote": {
            "available": True,
            "source": primary_source,
            "source_counts": dict(quote_sources),
            "updated_at": fetched_at,
            "interval_seconds": 120,
            "snapshot_count": len(quotes),
            "matched_count": matched_quotes,
            "used_count": used_realtime,
            "missing_history_count": missing_history,
            "history_gap_count": history_gap_count,
            "note": "批量实时行情只用于当前页面实时重算，不写回离线结果文件。",
        },
    })
    if pool == "hotmoney":
        base["market_regime"] = market_regime
    return base


def _score_ambush_candidates(
    conn: sqlite3.Connection,
    candidates: Sequence[Dict[str, Any]],
    as_of: Optional[str],
    pool: str,
) -> List[Dict[str, Any]]:
    """批量评分；每只候选按自身最后交易日获取 PIT 市场门。"""
    rows = [
        (
            cand,
            _recent_bars(
                conn, cand["code"],
                limit=PATTERN_EVAL_BARS,
                as_of=as_of,
            ),
        )
        for cand in candidates
    ]
    dates = [str(bars[-1].get("date") or "") for _cand, bars in rows if bars]
    p1_market_gates = (
        {} if pool == "etf" else _p1_market_gate_by_date(conn, dates)
    )
    return [
        _score_candidate_from_bars(
            cand,
            bars,
            pool=pool,
            p1_market_gate=p1_market_gates.get(
                str(bars[-1].get("date") or ""), False
            ) if bars else False,
        )
        for cand, bars in rows
    ]


def run_ambush(as_of: Optional[str] = None,
               max_cap: Optional[float] = None,
               pool: str = DEFAULT_POOL,
               write: bool = True,
               print_summary: bool = True) -> Dict[str, Any]:
    """write/print_summary=False 时只返回打分 payload、不落盘不打印（供 latent 模式复用打分核心）。"""
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        market_regime = _market_regime(conn, as_of) if pool == "hotmoney" else None
        scored = _score_ambush_candidates(conn, candidates, as_of, pool)
        if pool == "etf":
            capital_map, capital_available = {}, False
        else:
            capital_map, capital_available = _load_capital_map(conn, as_of)
    finally:
        conn.close()

    # 资金面维度：股东户数↓/回购=吸筹侧原始分、龙虎榜上榜=出货侧避雷。
    for r in scored:
        if pool != "etf":
            info = capital_map.get(r["code"])
            r["holder_change"] = info.get("holder_change") if info else None
            r["repurchase_recent"] = bool(info and info.get("repurchase_recent"))
            r["lhb_recent"] = bool(info and info.get("lhb_recent"))
        _apply_distribution_model(r, pool=pool)
        if pool != "etf":
            _attach_capital_evidence(r)

    # PIT 防泄漏：theme_candidates.json 是「最新日」快照，历史 as-of 复盘时用它会泄露未来题材热度，
    # 故只在 as_of=None（实时跑）时才挂题材/行业热度映射；历史复盘留空，避免未来热度泄漏。
    if pool == "etf":
        theme_map, sw2_heat_map = {}, {}
        theme_meta = {
            "available": False,
            "skipped_reason": "ETF 使用配置分类，不挂载个股申万行业/题材数据",
        }
    elif as_of is None:
        theme_map, theme_meta, sw2_heat_map = _load_theme_map()
    else:
        theme_map, sw2_heat_map = {}, {}
        theme_meta = {"available": False, "generated_at": None,
                      "age_days": None, "stale": False,
                      "skipped_reason": f"as_of={as_of} 历史复盘禁用题材缓存(防泄漏)"}
    for r in scored:
        sw2 = _clean_theme_name(r.get("parent_segment"))
        r["sw2_heat_pctile"] = sw2_heat_map.get(sw2)
        info = theme_map.get(r["code"])
        if info:
            r["tracking_theme"] = info["theme"]
            r["tracking_theme_code"] = info["theme_code"]
            r["theme_heat_pctile"] = info["heat_pctile"]

    ranked = [r for r in scored if r.get("ambush_score") is not None]
    _apply_accumulation_model(ranked, pool=pool)
    if pool == "hotmoney":
        _apply_reversal_model(ranked)              # 游资池主排序=超短反转分(纪要14)
        ranked.sort(key=lambda r: (r.get("reversal_score") or 0.0), reverse=True)
    else:
        ranked.sort(key=lambda r: (r.get("opportunity_score") or 0.0, r.get("ambush_score") or 0.0), reverse=True)
    skipped = len(scored) - len(ranked)

    accum_weights = accumulation_model_weights(pool)
    dist_weights = distribution_model_weights(pool)
    payload = base_payload("ambush", len(candidates), pool=pool)
    payload.update({
        "status": "ok" if candidates else "empty",
        "description": (
            "ETF 技术机会分：仅使用行情、量价、换手和技术形态；公司行为因子已剔除并对剩余权重重新归一。"
            if pool == "etf" else
            "细分龙头机会分：吸筹分、出货分展示原始模型分；仅计算机会分时把二者转为候选池百分位后做出货风险折扣。"
        ),
        "params": {
            "lookback": LOOKBACK,
            "cmf_full": CMF_FULL, "chip_band": CHIP_BAND, "pos_low": POS_LOW, "pos_high": POS_HIGH,
            "p1": _p1_rule_params(),
            "p4": _p4_rule_params(),
            "p5": _p5_rule_params(),
            "p15": _p15_rule_params(),
            "p16": _p16_rule_params(),
            "p20": _p20_rule_params(),
            "p21": _p21_rule_params(),
            "p26": _p26_rule_params(),
            "accumulation_model": {
                "weights": accum_weights,
                "features": list(accum_weights),
                "excluded_features": list(ETF_ACCUM_EXCLUDED) if pool == "etf" else [],
                "missing_holder_change_score": None if pool == "etf" else 50.0,
                "suspect_threshold": SUSPECT_ACCUM_SCORE,
                "suspect_rule": "patterns为空且吸筹分达到阈值",
                "experiment_file": display_path(ACCUM_EXPERIMENT_FILE),
            },
            "distribution_model": {
                "weights": dist_weights,
                "features": list(dist_weights),
                "excluded_features": list(ETF_DIST_EXCLUDED) if pool == "etf" else [],
                "warning_rule": distribution_warning_rule_metadata(),
                "experiment_file": display_path(DIST_EXPERIMENT_FILE),
            },
            "opportunity_model": {
                "formula": OPPORTUNITY_FORMULA,
                "distribution_penalty": OPPORTUNITY_DISTRIBUTION_PENALTY,
                "description": "机会分 = 吸筹百分位 × (1 - 0.5 × 出货百分位 / 100)，出货最多折掉 50%。",
            },
        },
        "as_of": as_of,
        "pool": pool,
        "asset_type": "etf" if pool == "etf" else "stock",
        "analysis_scope": (
            "technical_heuristic_unvalidated_for_etf"
            if pool == "etf" else "validated_stock_pool"
        ),
        "market_regime": market_regime,
        "reversal_model": ({"weights": REVERSAL_WEIGHTS, "features": list(REVERSAL_FEATURES),
                            "smooth_days": REVERSAL_SMOOTH_DAYS,
                            "smooth": "ema" if REVERSAL_SMOOTH_DAYS > 1 else "snapshot",
                            "note": "游资池主排序分；过热因子反向加权=超短反转(纪要14)；"
                                    "近N日截面子分EMA平滑(纪要15,favorable/近段更稳)"}
                           if pool == "hotmoney" else None),
        "scored_count": len(ranked),
        "skipped_count": skipped,
        "triggered_count": sum(1 for r in ranked if r.get("triggered")),
        "phase_counts": dict(Counter(r.get("pattern_phase") for r in ranked)),
        "max_market_cap_yi": None if pool == "etf" else max_cap,
        "theme_source": theme_meta,
        "capital_applicable": pool != "etf",
        "capital_available": capital_available,
        "company_data_skipped": (
            ["financials", "valuation", "pledge", "shareholder_count", "repurchase", "lhb"]
            if pool == "etf" else []
        ),
        "capital_counts": {
            "holder_down": sum(1 for r in ranked if (r.get("holder_change") or 0) < 0),
            "repurchase": sum(1 for r in ranked if r.get("repurchase_recent")),
            "lhb_avoid": sum(1 for r in ranked if r.get("lhb_recent")),
        },
        "stocks": ranked,
    })
    if pool == "etf":
        payload["notes"] = [
            "ETF 使用技术观察口径；P1-P26 的历史有效性来自股票双池，尚未做 ETF 专项回测。",
            "已跳过财报、估值、质押、股东户数、公司回购、龙虎榜和个股题材数据。",
        ]
    if not candidates:
        payload["notes"] = (["ETF 配置为空：请编辑 stock_etf_pool.py 中的 fund_index。"]
                            if pool == "etf" else
                            ["候选池为空：先运行 python stock_crawl_segment_leaders.py crawl 选龙头并回写 is_leader。"])
    if write:
        write_payload(AMBUSH_RESULT_FILE, payload)
    if print_summary:
        _print_ambush_summary(payload)
    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def _disp_width(text: str) -> int:
    """终端显示宽度：东亚全角/宽字符（含 emoji）按 2 计，其余按 1。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(text))


def _ljust(text: Any, width: int) -> str:
    """按显示宽度左对齐（中文每字算 2 宽，避免列错位）。"""
    text = str(text)
    return text + " " * max(0, width - _disp_width(text))


def _rjust(text: Any, width: int) -> str:
    """按显示宽度右对齐。"""
    text = str(text)
    return " " * max(0, width - _disp_width(text)) + text


def _theme_cell(s: Dict[str, Any]) -> str:
    """走势相似行业 + 行业热度百分位（来自 stock_theme_candidates.py）。

    有题材映射时显示「行业名 热度NN%」；缺映射时显示空值，真实 SW2 另列展示。
    """
    theme = s.get("tracking_theme")
    if not theme:
        return ""
    pctile = s.get("theme_heat_pctile")
    return theme if pctile is None else f"{theme} 热度{pctile:.0f}%"


def _sw2_cell(s: Dict[str, Any]) -> str:
    industry = _clean_theme_name(s.get("parent_segment"))
    if not industry:
        return "-"
    pctile = s.get("sw2_heat_pctile")
    return industry if pctile is None else f"{industry} 热度{pctile:.0f}%"


def _theme_freshness_note(meta: Dict[str, Any]) -> str:
    """题材热度数据来源 + 新鲜度提示行。"""
    if meta.get("skipped_reason"):
        return f"题材热度: ⏸ {meta['skipped_reason']}，「走势相似行业」列置空"
    if not meta.get("available"):
        return ("题材热度: ⚠️ 缺 theme_candidates.json，"
                "「走势相似行业」列置空 → 先跑 python stock_theme_candidates.py")
    gen = meta.get("generated_at") or "?"
    age = meta.get("age_days")
    if age is None:
        return f"题材热度: 来自 stock_theme_candidates.py（生成于 {gen}）"
    freshness = f"{age} 天前生成"
    if meta.get("stale"):
        return (f"题材热度: ⚠️ 数据偏旧（{freshness}，≥{THEME_STALE_DAYS}天）"
                f" → 重跑 python stock_theme_candidates.py 刷新热度")
    return f"题材热度: 来自 stock_theme_candidates.py（{freshness}，{gen}）"


def _print_ambush_summary(payload: Dict[str, Any]) -> None:
    stocks = payload.get("stocks", [])
    print("=" * 112)
    print("  主力资金雷达 · 吸筹分 + 游资形态 (ambush)")
    counts = payload.get("phase_counts") or {}
    dist = " · ".join(f"{ph}{counts[ph]}" for ph in PHASE_ORDER if counts.get(ph))
    as_of_note = f" · 数据截至 {payload['as_of']}(历史复盘·禁题材缓存)" if payload.get("as_of") else ""
    cap = payload.get("max_market_cap_yi")
    if payload.get("pool") == "etf":
        pool_note = "ETF配置池·技术口径"
    elif payload.get("pool") == "hotmoney":
        pool_note = "游资小盘池"
    else:
        pool_note = f"≤{cap:g}亿小中盘龙头" if cap else "全市值龙头(含大盘)"
    print(f"  生成时间: {payload['generated_at']}{as_of_note} · 候选({pool_note}): "
          f"{payload['candidate_count']} · 已打分: {payload.get('scored_count', 0)}")
    print(f"  阶段分布(游资操作顺序): {dist}")
    cc = payload.get("capital_counts") or {}
    if payload.get("capital_available"):
        print(f"  资金面(纪要13): 户数下降{cc.get('holder_down',0)} · 回购中{cc.get('repurchase',0)}"
              f" · 近期龙虎榜波动提示{cc.get('lhb_avoid',0)} · 户数/回购已并入吸筹总分 · 排序=机会分")
    print(f"  落盘: {display_path(AMBUSH_RESULT_FILE)}")
    print(f"  {_theme_freshness_note(payload.get('theme_source') or {})}")
    print("-" * 112)
    if not stocks:
        for note in payload.get("notes", ["（无候选）"]):
            print(f"  {note}")
        print("=" * 112)
        return
    print(f"  {'#':>2} {_ljust('代码', 7)}{_ljust('名称', 9)}"
          f"{_rjust('机会分', 7)}{_rjust('吸筹分', 7)}{_rjust('出货分', 7)}  "
          f"{_rjust('量比', 5)} {_rjust('价分位', 6)} {_rjust('换手分位', 8)} {'CMF':>6} {_rjust('筹码集中', 7)}"
          f" {_rjust('连板', 4)}  {_ljust('命中形态', 16)} 阶段 / 走势相似行业(热度%) / 二级行业(热度%)")
    for i, s in enumerate(stocks[:30], 1):
        sig = s.get("signals", {})
        name = (s.get("name") or "")[:8]
        pats = ",".join(s.get("patterns") or []) or "-"
        phase = s.get("pattern_phase") or s.get("state", "")
        conf = s.get("phase_confidence")
        phase_cell = f"{phase} 把握{conf:.0f}" if conf is not None else phase
        tp = sig.get("turnover_pctile")
        tp_cell = "-" if tp is None else (f"{tp:.2f}!" if tp > 0.84 else f"{tp:.2f}")  # !=拥挤(>84%)
        acc_raw = s.get("ambush_score", 0)
        dist_raw = s.get("distribution_score", 0)
        print(f"  {i:>2} {s['code']:<7}{_ljust(name, 9)}"
              f"{(s.get('opportunity_score') or 0):>7.1f}{(acc_raw or 0):>7.1f}{(dist_raw or 0):>7.1f}  "
              f"{_fmt(sig.get('vol_ratio')):>5} {_fmt(sig.get('close_pctile')):>6} {tp_cell:>8} "
              f"{_fmt(sig.get('cmf')):>6} {_fmt(sig.get('chip_concentration')):>7} "
              f"{sig.get('limit_streak', 0):>4}  {pats:<16} {phase_cell} / "
              f"{_theme_cell(s) or '-'} / {_sw2_cell(s)}")
    if len(stocks) > 30:
        print(f"  ... 其余 {len(stocks) - 30} 只见 {display_path(AMBUSH_RESULT_FILE)}")
    print("=" * 112)


# ── verify：潜伏分后验回测 ────────────────────────────────────

def _collect_verify_samples(conn: sqlite3.Connection, candidates: List[Dict[str, Any]],
                            pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    """对每只候选滑动取历史截面，PIT 重算十因子吸筹分并配对未来前向收益。

    返回 {samples, dates, codes}。samples 每项 = {date, code, score, rets:{h:ret}}。
    PIT：打分只用截止 as-of 当日的 LOOKBACK 根 bar；前向收益用其后第 h 根 bar 的收盘价。
    """
    max_h = max(VERIFY_HORIZONS)
    series: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, int]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})

    if not series:
        return {"samples": [], "dates": [], "codes": []}

    histories = _load_capital_histories(conn)
    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    usable = all_dates[:-max_h]                       # 末段没有前向数据，剔除
    as_of_dates = _verify_as_of_dates(usable)
    p1_market_gates = _p1_market_gate_by_date(conn, as_of_dates)
    cutoffs = {
        d: (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=CAPITAL_EVENT_DAYS)).strftime("%Y-%m-%d")
        for d in as_of_dates
    }

    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    for d in as_of_dates:
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK - 1 or i + max_h >= len(bars):
                continue
            window = bars[i - LOOKBACK + 1:i + 1]
            res = _score_bars(code, window)
            if res is None:
                continue
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in VERIFY_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            fired = match_patterns(
                code, window, pool=pool,
                p1_market_gate=p1_market_gates.get(d, False),
            )
            feature_row = {
                "sub_scores": res.get("sub_scores") or {},
                "patterns": [pattern["code"] for pattern in fired],
                "holder_change": _holder_change_at(histories, code, d),
                "repurchase_recent": _recent_date_hit(
                    histories["repurchase_dates"], code, d, cutoffs[d]
                ),
            }
            score = _accumulation_model_score(_accumulation_raw_features(feature_row))
            samples.append({"date": d, "code": code, "score": score,
                            "triggered": res["triggered"], "rets": rets})
            used_dates.add(d)
    return {"samples": samples, "dates": sorted(used_dates), "codes": list(series.keys())}


def _aggregate_verify(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分位前向收益（pooled）+ 截面 Rank IC + 多空分位价差（按 as-of 日聚合再平均）。"""
    by_horizon: Dict[str, Any] = {}
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_date.setdefault(s["date"], []).append(s)

    for h in VERIFY_HORIZONS:
        # pooled：全样本按潜伏分五等分，看各桶平均前向收益（绝对收益含大盘 beta）
        ordered = sorted(samples, key=lambda s: s["score"])
        n = len(ordered)
        buckets = []
        for q in range(VERIFY_BUCKETS):
            grp = ordered[q * n // VERIFY_BUCKETS:(q + 1) * n // VERIFY_BUCKETS]
            rets = [g["rets"][h] for g in grp]
            buckets.append({
                "mean_ret": round(_mean(rets), 4) if rets else None,
                "n": len(grp),
                "score_lo": round(grp[0]["score"], 1) if grp else None,
                "score_hi": round(grp[-1]["score"], 1) if grp else None,
            })
        # 截面：每个 as-of 日单独算 Rank IC 与 top-bottom 五分位价差，再跨日平均（剔除大盘 beta）
        ics: List[float] = []
        spreads: List[float] = []
        for grp in by_date.values():
            if len(grp) < VERIFY_MIN_NAMES:
                continue
            scores = [g["score"] for g in grp]
            rets = [g["rets"][h] for g in grp]
            ic = _spearman(scores, rets)
            if ic is not None:
                ics.append(ic)
            sg = sorted(grp, key=lambda g: g["score"])
            k = max(1, len(sg) // VERIFY_BUCKETS)
            top = _mean([g["rets"][h] for g in sg[-k:]])
            bot = _mean([g["rets"][h] for g in sg[:k]])
            if top is not None and bot is not None:
                spreads.append(top - bot)

        ic_mean = _mean(ics)
        ic_std = (sum((x - ic_mean) ** 2 for x in ics) / len(ics)) ** 0.5 if len(ics) > 1 else None
        t_stat = (ic_mean / ic_std * math.sqrt(len(ics))) if ic_mean is not None and ic_std else None
        pooled_hi, pooled_lo = buckets[-1]["mean_ret"], buckets[0]["mean_ret"]
        by_horizon[str(h)] = {
            "quantile_returns": buckets,
            "pooled_q5_minus_q1": round(pooled_hi - pooled_lo, 4) if pooled_hi is not None and pooled_lo is not None else None,
            "long_short_spread": round(_mean(spreads), 4) if spreads else None,
            "ic_mean": round(ic_mean, 4) if ic_mean is not None else None,
            "ic_std": round(ic_std, 4) if ic_std is not None else None,
            "ic_t_stat": round(t_stat, 2) if t_stat is not None else None,
            "ic_hit_rate": round(sum(1 for x in ics if x > 0) / len(ics), 3) if ics else None,
            "n_sections": len(ics),
            "trigger": _trigger_study(by_date, h),
        }
    return by_horizon


def _trigger_study(by_date: Dict[str, List[Dict[str, Any]]], h: int) -> Dict[str, Any]:
    """右进左出事件研究：触发样本 vs 全体的前向收益。

    excess = 每个 as-of 日「触发样本均收益 − 该日全样本均收益」(剔除大盘 beta)，跨日平均 + t 值。
    这才是右进左出的真实择时力——触发那一刻买，能否跑赢同日普通龙头。
    """
    excess: List[float] = []
    trig_rets: List[float] = []
    base_rets: List[float] = []
    for grp in by_date.values():
        fired = [g["rets"][h] for g in grp if g.get("triggered")]
        if not fired:
            continue
        section_mean = _mean([g["rets"][h] for g in grp])
        trig_mean = _mean(fired)
        excess.append(trig_mean - section_mean)
        trig_rets.extend(fired)
        base_rets.extend(g["rets"][h] for g in grp)
    if not excess:
        return {"n_triggered": 0, "n_sections": 0}
    em = _mean(excess)
    es = (sum((x - em) ** 2 for x in excess) / len(excess)) ** 0.5 if len(excess) > 1 else None
    t = (em / es * math.sqrt(len(excess))) if es else None
    return {
        "n_triggered": len(trig_rets),
        "n_sections": len(excess),
        "triggered_mean_ret": round(_mean(trig_rets), 4),
        "all_mean_ret": round(_mean(base_rets), 4),
        "excess_mean": round(em, 4),
        "excess_t_stat": round(t, 2) if t is not None else None,
        "win_rate": round(sum(1 for x in trig_rets if x > 0) / len(trig_rets), 3),
    }


def _verdict(by_horizon: Dict[str, Any]) -> str:
    mid = by_horizon.get(str(VERIFY_HORIZONS[len(VERIFY_HORIZONS) // 2]), {})
    ic, t = mid.get("ic_mean"), mid.get("ic_t_stat")
    tg = mid.get("trigger", {})
    ex, et = tg.get("excess_mean"), tg.get("excess_t_stat")
    parts = []
    # 触发器(右进左出)是主信号：触发那刻买能否跑赢同日龙头
    if ex is not None and et is not None:
        if ex > 0 and et >= 2:
            parts.append(f"右进左出触发器有效：触发后超额 {_pct(ex).strip()} (t={et}, 显著)")
        elif ex > 0 and et >= 1:
            parts.append(f"触发器方向为正但偏弱：超额 {_pct(ex).strip()} (t={et})")
        elif ex <= 0:
            parts.append(f"触发器无超额：{_pct(ex).strip()} (t={et})")
    if ic is None:
        return "; ".join(parts) or "样本不足，无法判定"
    if ic >= 0.03 and (t or 0) >= 2:
        parts.append(f"吸筹分截面 IC 显著正 ({ic}, t={t})")
    elif ic > 0.005:
        parts.append(f"吸筹分截面 IC 微正 ({ic}, t={t})")
    elif ic <= -0.01:
        parts.append(f"吸筹分截面 IC 偏负 ({ic}, t={t})")
    else:
        parts.append(f"吸筹分截面 IC 中性 ({ic}, t={t})")
    return "; ".join(parts)


def run_verify(max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_verify_samples(conn, candidates, pool=pool)
    finally:
        conn.close()

    samples = collected["samples"]
    by_horizon = _aggregate_verify(samples) if samples else {}
    payload = base_payload("verify", len(candidates), pool=pool)
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "十因子吸筹分后验回测：PIT 重算吸筹分 vs 未来前向收益（分位单调性 / 截面RankIC / 多空价差）。",
        "params": {
            "horizons": list(VERIFY_HORIZONS), "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS, "buckets": VERIFY_BUCKETS,
            "min_names_per_section": VERIFY_MIN_NAMES,
        },
        "section_count": len(collected["dates"]),
        "sample_count": len(samples),
        "scored_codes": len(collected["codes"]),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "horizons": by_horizon,
        "verdict": _verdict(by_horizon) if samples else "候选池为空或历史不足，无法回测。",
    })
    if not samples:
        payload["notes"] = ["无可回测样本：先确保 sw3_member.is_leader 有龙头、且其历史日线足够长。"]
    write_payload(VERIFY_RESULT_FILE, payload)
    _print_verify_summary(payload)
    return payload


def _pct(value: Any) -> str:
    return "  -  " if value is None else f"{value * 100:+5.2f}%"


def _print_verify_summary(payload: Dict[str, Any]) -> None:
    print("=" * 92)
    print("  主力资金雷达 · 潜伏分后验回测 (verify)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 候选龙头: {payload['candidate_count']}"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {display_path(VERIFY_RESULT_FILE)}")
    print("-" * 92)
    horizons = payload.get("horizons", {})
    if not horizons:
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 92)
        return
    print("  分位前向收益（按潜伏分五等分，Q1低→Q5高；绝对收益含大盘 beta，pooled）:")
    print(f"  {'持有期':>6} {'Q1低':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Q5高':>8} | {'Q5-Q1':>8}")
    for h in VERIFY_HORIZONS:
        hz = horizons.get(str(h), {})
        b = hz.get("quantile_returns", [])
        cells = " ".join(f"{_pct(b[q]['mean_ret']) if q < len(b) else '  -  ':>8}" for q in range(VERIFY_BUCKETS))
        print(f"  {str(h)+'日':>6} {cells} | {_pct(hz.get('pooled_q5_minus_q1')):>8}")
    print("-" * 92)
    print("  截面口径（每个 as-of 日内部排序，跨日平均；天然剔除大盘 beta）— 这才是真实选股力:")
    print(f"  {'持有期':>6} {'RankIC':>9} {'IC_t':>7} {'胜率':>7} {'多空价差':>9} {'截面数':>7}")
    for h in VERIFY_HORIZONS:
        hz = horizons.get(str(h), {})
        hit = hz.get("ic_hit_rate")
        print(f"  {str(h)+'日':>6} {_fmt(hz.get('ic_mean')):>9} {_fmt(hz.get('ic_t_stat')):>7} "
              f"{(f'{hit*100:.0f}%' if hit is not None else '-'):>7} {_pct(hz.get('long_short_spread')):>9} "
              f"{hz.get('n_sections', 0):>7}")
    print("-" * 92)
    tg0 = horizons.get(str(VERIFY_HORIZONS[0]), {}).get("trigger", {})
    print(f"  右进左出触发器事件研究（触发=吸筹中放量突破近端高点；触发样本数 {tg0.get('n_triggered', 0)}）:")
    print(f"  {'持有期':>6} {'触发后均收益':>11} {'同日全体':>9} {'超额':>9} {'超额_t':>7} {'胜率':>7}")
    for h in VERIFY_HORIZONS:
        tg = horizons.get(str(h), {}).get("trigger", {})
        win = tg.get("win_rate")
        print(f"  {str(h)+'日':>6} {_pct(tg.get('triggered_mean_ret')):>11} {_pct(tg.get('all_mean_ret')):>9} "
              f"{_pct(tg.get('excess_mean')):>9} {_fmt(tg.get('excess_t_stat')):>7} "
              f"{(f'{win*100:.0f}%' if win is not None else '-'):>7}")
    print("-" * 92)
    print(f"  结论: {payload.get('verdict')}")
    print("=" * 92)


# ── patterns：形态预测力后验（PIT 事件研究）─────────────────────

def _collect_pattern_samples(conn: sqlite3.Connection, candidates: List[Dict[str, Any]],
                             pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    """对每只龙头滑动取历史截面，PIT 匹配全部形态并配对未来前向收益。

    samples 每项 = {date, fired:set(形态code), rets:{h:ret}}。PIT 安全：形态只用截止当日窗口。
    """
    max_h = max(VERIFY_HORIZONS)
    series: Dict[str, List[Dict[str, Any]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + CHIP_WINNER_RISK_DAYS + max_h + 1:
            continue
        series[cand["code"]] = bars
    if not series:
        return {"samples": [], "dates": [], "codes": []}

    all_dates = sorted({b["date"] for bars in series.values() for b in bars})
    as_of_dates = _verify_as_of_dates(all_dates[:-max_h])
    as_of_set = set(as_of_dates)
    p1_market_gates = _p1_market_gate_by_date(conn, as_of_dates)
    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    first_index = LOOKBACK + CHIP_WINNER_RISK_DAYS - 1
    for code, bars in series.items():
        # P26 使用的“5 日前筹码”与同股 i-5 锚点的当前筹码完全相同。
        # 按股票顺序扫描可直接复用，逐日全量回测时把筹码重建从每锚点2次降到约1次。
        chip_by_index: Dict[int, Optional[Dict[str, float]]] = {}
        for i in range(first_index, len(bars) - max_h):
            d = bars[i]["date"]
            if d not in as_of_set:
                continue
            window = bars[max(0, i - PATTERN_EVAL_BARS + 1):i + 1]
            cached_prior = chip_by_index.get(i - CHIP_WINNER_RISK_DAYS, _PRIOR_CHIP_UNSET)
            ctx = _build_pattern_context(
                code,
                window,
                prior_chip=cached_prior,
                defer_chip=True,
            )
            fired = match_patterns(
                code, window, ctx=ctx, pool=pool,
                p1_market_gate=p1_market_gates.get(d, False),
            )
            if ctx.get("_chip_computed"):
                chip_by_index[i] = ctx.get("chip")
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in VERIFY_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            samples.append({"date": d, "fired": {p["code"] for p in fired}, "rets": rets})
            used_dates.add(d)
    return {"samples": samples, "dates": sorted(used_dates), "codes": list(series.keys())}


def _pattern_date_plan(
    conn: sqlite3.Connection,
    candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    """用轻量 SQL 生成形态回测日期与工作量计划，避免父进程加载全部日线。"""
    codes = list(dict.fromkeys(str(candidate["code"]) for candidate in candidates))
    min_bars = LOOKBACK + CHIP_WINNER_RISK_DAYS + max(VERIFY_HORIZONS) + 1
    counts: Dict[str, int] = {}
    for start in range(0, len(codes), 500):
        chunk = codes[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT code, COUNT(*) AS n FROM stock_history "
            f"WHERE code IN ({placeholders}) AND daily_close IS NOT NULL "
            f"AND daily_volume IS NOT NULL GROUP BY code",
            chunk,
        ).fetchall()
        counts.update({str(row["code"]): int(row["n"]) for row in rows})

    eligible = [candidate for candidate in candidates if counts.get(str(candidate["code"]), 0) >= min_bars]
    eligible_codes = [str(candidate["code"]) for candidate in eligible]
    dates: set = set()
    for start in range(0, len(eligible_codes), 500):
        chunk = eligible_codes[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT DISTINCT date FROM stock_history WHERE code IN ({placeholders}) "
            f"AND daily_close IS NOT NULL AND daily_volume IS NOT NULL",
            chunk,
        ).fetchall()
        dates.update(str(row["date"]) for row in rows)
    ordered_dates = sorted(dates)
    as_of_dates = _verify_as_of_dates(ordered_dates[:-max(VERIFY_HORIZONS)])
    anchor_counts = {
        code: max(0, counts.get(code, 0) - (LOOKBACK + CHIP_WINNER_RISK_DAYS - 1) - max(VERIFY_HORIZONS))
        for code in eligible_codes
    }
    return eligible, as_of_dates, anchor_counts


def _pattern_aggregate_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """按股票分片回测并在进程内聚合，避免跨进程传输逐股票日明细。"""
    db_file, candidates, pool, as_of_dates, pattern_codes = task
    selected_codes = set(pattern_codes)
    selected_patterns = [pattern for pattern in PATTERNS if pattern[0] in selected_codes]
    horizon_count = len(VERIFY_HORIZONS)
    max_h = max(VERIFY_HORIZONS)
    as_of_set = set(as_of_dates)
    first_index = LOOKBACK + CHIP_WINNER_RISK_DAYS - 1
    by_date: Dict[str, Any] = {}
    sample_count = 0
    scored_codes = 0

    conn = stock_storage.connect(db_file)
    try:
        p1_market_gates = _p1_market_gate_by_date(conn, as_of_dates)
        for candidate in candidates:
            code = str(candidate["code"])
            bars = _all_bars(conn, code)
            if len(bars) < LOOKBACK + CHIP_WINNER_RISK_DAYS + max_h + 1:
                continue
            scored_codes += 1
            chip_by_index: Dict[int, Optional[Dict[str, float]]] = {}
            for index in range(first_index, len(bars) - max_h):
                date = bars[index]["date"]
                if date not in as_of_set:
                    continue
                window = bars[max(0, index - PATTERN_EVAL_BARS + 1):index + 1]
                cached_prior = chip_by_index.get(index - CHIP_WINNER_RISK_DAYS, _PRIOR_CHIP_UNSET)
                context = _build_pattern_context(
                    code,
                    window,
                    prior_chip=cached_prior,
                    defer_chip=True,
                )
                fired = match_patterns(
                    code,
                    window,
                    ctx=context,
                    pool=pool,
                    p1_market_gate=p1_market_gates.get(date, False),
                    patterns=selected_patterns,
                )
                if context.get("_chip_computed"):
                    chip_by_index[index] = context.get("chip")
                close = bars[index]["close"]
                if not close:
                    continue
                future_closes = [bars[index + horizon]["close"] for horizon in VERIFY_HORIZONS]
                if any(not value for value in future_closes):
                    continue
                returns = [float(value) / close - 1.0 for value in future_closes]
                if any(not math.isfinite(value) for value in returns):
                    continue

                # [样本数, 各周期收益和, {形态: [命中数, 各周期收益和, 各周期胜数]}]
                daily = by_date.setdefault(date, [0, [0.0] * horizon_count, {}])
                daily[0] += 1
                sample_count += 1
                for horizon_index, value in enumerate(returns):
                    daily[1][horizon_index] += value
                for pattern in fired:
                    hit = daily[2].setdefault(
                        pattern["code"],
                        [0, [0.0] * horizon_count, [0] * horizon_count],
                    )
                    hit[0] += 1
                    for horizon_index, value in enumerate(returns):
                        hit[1][horizon_index] += value
                        if value > 0.0:
                            hit[2][horizon_index] += 1
    finally:
        conn.close()
    return {
        "by_date": by_date,
        "sample_count": sample_count,
        "scored_codes": scored_codes,
    }


def _merge_pattern_aggregates(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    horizon_count = len(VERIFY_HORIZONS)
    merged: Dict[str, Any] = {}
    sample_count = 0
    scored_codes = 0
    for part in parts:
        sample_count += int(part.get("sample_count", 0))
        scored_codes += int(part.get("scored_codes", 0))
        for date, source in part.get("by_date", {}).items():
            target = merged.setdefault(date, [0, [0.0] * horizon_count, {}])
            target[0] += source[0]
            for horizon_index in range(horizon_count):
                target[1][horizon_index] += source[1][horizon_index]
            for code, source_hit in source[2].items():
                target_hit = target[2].setdefault(
                    code,
                    [0, [0.0] * horizon_count, [0] * horizon_count],
                )
                target_hit[0] += source_hit[0]
                for horizon_index in range(horizon_count):
                    target_hit[1][horizon_index] += source_hit[1][horizon_index]
                    target_hit[2][horizon_index] += source_hit[2][horizon_index]
    return {
        "by_date": merged,
        "dates": sorted(merged),
        "sample_count": sample_count,
        "scored_codes": scored_codes,
    }


def _collect_pattern_aggregates(
    conn: sqlite3.Connection,
    candidates: List[Dict[str, Any]],
    pool: str,
    pattern_codes: Sequence[str],
    jobs: int = 1,
) -> Dict[str, Any]:
    """并行收集形态事件的日期级充分统计量，结果与逐条样本事件研究等价。"""
    eligible, as_of_dates, anchor_counts = _pattern_date_plan(conn, candidates)
    if not eligible or not as_of_dates:
        return {"by_date": {}, "dates": [], "sample_count": 0, "scored_codes": 0, "worker_count": 0}

    worker_count = max(1, min(int(jobs), len(eligible)))
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for candidate in sorted(eligible, key=lambda row: anchor_counts.get(str(row["code"]), 0), reverse=True):
        bucket_index = min(range(worker_count), key=loads.__getitem__)
        buckets[bucket_index].append(candidate)
        loads[bucket_index] += anchor_counts.get(str(candidate["code"]), 0)
    tasks = [
        (str(DB_FILE), bucket, pool, as_of_dates, list(pattern_codes))
        for bucket in buckets
        if bucket
    ]
    if len(tasks) == 1:
        parts = [_pattern_aggregate_worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            parts = list(executor.map(_pattern_aggregate_worker, tasks))
    result = _merge_pattern_aggregates(parts)
    result["worker_count"] = len(tasks)
    return result


def _horizon_effective(signal: str, hz: Dict[str, Any]) -> bool:
    """单个(形态,周期)是否有效；observe只留统计，不可自动升级为交易信号。"""
    em, t = hz.get("excess_mean"), hz.get("excess_t_stat")
    if em is None or t is None:
        return False
    if signal == "observe":
        return False
    if signal == "buy":
        return em > 0 and t >= 1.5
    if signal == "sell":
        return em < 0 and t <= -1.5
    return abs(t) >= 1.5      # hold(试盘/突破/拉升)无先验方向，显著即"有信息"


def _pattern_event_study(by_date: Dict[str, List[Dict[str, Any]]], pcode: str, h: int) -> Dict[str, Any]:
    """单形态事件研究：命中样本 vs 同日全体的前向收益超额（剔除大盘 beta），跨日平均 + t。"""
    excess: List[float] = []
    hit_rets: List[float] = []
    for date in sorted(by_date):
        grp = by_date[date]
        hit = [g["rets"][h] for g in grp if pcode in g["fired"]]
        if not hit:
            continue
        section_mean = _mean([g["rets"][h] for g in grp])
        excess.append(_mean(hit) - section_mean)
        hit_rets.extend(hit)
    if not hit_rets:
        return {
            "n_hits": 0, "n_sections": 0, "excess_mean": None,
            "excess_t_stat": None, "excess_p_value": None,
            "hac_lag": max(0, math.ceil(h / max(1, VERIFY_STEP)) - 1),
            "win_rate": None,
        }
    em = _mean(excess)
    lag = max(0, math.ceil(h / max(1, VERIFY_STEP)) - 1)
    t, p_value = _newey_west_mean_test(excess, lag)
    return {
        "n_hits": len(hit_rets),
        "n_sections": len(excess),
        "excess_mean": round(em, 4),
        "excess_t_stat": round(t, 2) if t is not None else None,
        "excess_p_value": round(p_value, 6) if p_value is not None else None,
        "hac_lag": lag,
        "win_rate": round(sum(1 for x in hit_rets if x > 0) / len(hit_rets), 3),
    }


def _pattern_event_study_aggregated(
    by_date: Dict[str, Any],
    pattern_code: str,
    horizon: int,
) -> Dict[str, Any]:
    """从日期级充分统计量还原事件研究，避免保留数百万条逐股票日样本。"""
    horizon_index = VERIFY_HORIZONS.index(horizon)
    excess: List[float] = []
    hit_count = 0
    win_count = 0
    for date in sorted(by_date):
        daily = by_date[date]
        hit = daily[2].get(pattern_code)
        if not hit or not hit[0] or not daily[0]:
            continue
        excess.append(hit[1][horizon_index] / hit[0] - daily[1][horizon_index] / daily[0])
        hit_count += int(hit[0])
        win_count += int(hit[2][horizon_index])
    if not hit_count:
        return {
            "n_hits": 0,
            "n_sections": 0,
            "excess_mean": None,
            "excess_t_stat": None,
            "excess_p_value": None,
            "hac_lag": max(0, math.ceil(horizon / max(1, VERIFY_STEP)) - 1),
            "win_rate": None,
        }
    excess_mean = _mean(excess)
    lag = max(0, math.ceil(horizon / max(1, VERIFY_STEP)) - 1)
    t_stat, p_value = _newey_west_mean_test(excess, lag)
    return {
        "n_hits": hit_count,
        "n_sections": len(excess),
        "excess_mean": round(excess_mean, 4),
        "excess_t_stat": round(t_stat, 2) if t_stat is not None else None,
        "excess_p_value": round(p_value, 6) if p_value is not None else None,
        "hac_lag": lag,
        "win_rate": round(win_count / hit_count, 3),
    }


def _newey_west_mean_test(
    values: Sequence[float], lag: int
) -> Tuple[Optional[float], Optional[float]]:
    """Newey-West/Bartlett mean test for overlapping forward returns."""
    n = len(values)
    if n < 2:
        return None, None
    mean = float(sum(values) / n)
    centered = [float(value) - mean for value in values]
    long_variance = sum(value * value for value in centered) / n
    max_lag = min(max(0, int(lag)), n - 1)
    for offset in range(1, max_lag + 1):
        covariance = sum(
            centered[index] * centered[index - offset]
            for index in range(offset, n)
        ) / n
        long_variance += 2.0 * (1.0 - offset / (max_lag + 1.0)) * covariance
    if long_variance <= 0.0:
        return None, None
    t_stat = mean / math.sqrt(long_variance / n)
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return t_stat, p_value


def _bh_fdr_threshold(values: Sequence[float], q: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    accepted = None
    for rank, value in enumerate(ordered, 1):
        if value <= q * rank / count:
            accepted = value
    return accepted


def run_patterns(
    max_cap: Optional[float] = None,
    pool: str = DEFAULT_POOL,
    jobs: int = 1,
    pattern_min: int = 1,
    pattern_max: int = 26,
) -> Dict[str, Any]:
    pattern_min = max(1, min(26, int(pattern_min)))
    pattern_max = max(1, min(26, int(pattern_max)))
    if pattern_min > pattern_max:
        pattern_min, pattern_max = pattern_max, pattern_min
    selected_patterns = [
        pattern for pattern in PATTERNS
        if pattern_min <= int(pattern[0][1:]) <= pattern_max
    ]
    pattern_codes = [pattern[0] for pattern in selected_patterns]
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_pattern_aggregates(
            conn,
            candidates,
            pool=pool,
            pattern_codes=pattern_codes,
            jobs=jobs,
        )
    finally:
        conn.close()

    by_date = collected["by_date"]

    results = []
    p_values: List[float] = []
    for pcode, name, phase, signal, _fn in selected_patterns:
        horizons = {
            str(h): _pattern_event_study_aggregated(by_date, pcode, h)
            for h in VERIFY_HORIZONS
        }
        p_values.extend(
            horizons[str(h)].get("excess_p_value")
            if horizons[str(h)].get("excess_p_value") is not None else 1.0
            for h in VERIFY_HORIZONS
        )
        row = {"code": pcode, "name": name, "phase": phase, "signal": signal,
               "horizons": horizons}
        results.append(row)

    fdr_threshold = _bh_fdr_threshold(p_values, PATTERN_FDR_Q)
    for row in results:
        effective_at = []
        for h in VERIFY_HORIZONS:
            horizon = row["horizons"][str(h)]
            p_value = horizon.get("excess_p_value")
            horizon["fdr_10"] = bool(
                fdr_threshold is not None
                and p_value is not None
                and p_value <= fdr_threshold
            )
            if horizon["fdr_10"] and _horizon_effective(row["signal"], horizon):
                effective_at.append(h)
        row["effective_at"] = effective_at

    payload = base_payload("patterns", len(candidates), pool=pool)
    payload.update({
        "status": "ok" if collected["sample_count"] else "empty",
        "pool": pool,
        "description": "游资形态预测力后验：每个形态命中后，未来 N 日相对同日全体的超额收益（剔除大盘 beta）；重叠收益使用 Newey-West HAC，并控制多重检验。",
        "params": {
            "horizons": list(VERIFY_HORIZONS),
            "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS,
            "pattern_codes": pattern_codes,
            "worker_count": collected.get("worker_count", 1),
            "p1": _p1_rule_params(),
            "p4": _p4_rule_params(),
            "p5": _p5_rule_params(),
            "p15": _p15_rule_params(),
            "p16": _p16_rule_params(),
            "p17": _p17_rule_params(),
            "p21": _p21_rule_params(),
            "p23": _p23_rule_params(),
            "p25": _p25_rule_params(),
            "p26": _p26_rule_params(),
            "significance": "Newey-West/Bartlett HAC; lag=ceil(horizon/step)-1",
            "multiple_testing": f"Benjamini-Hochberg FDR {PATTERN_FDR_Q:.0%} across patterns x horizons",
        },
        "bh_fdr_q": PATTERN_FDR_Q,
        "bh_fdr_p_threshold": round(fdr_threshold, 6) if fdr_threshold is not None else None,
        "section_count": len(collected["dates"]),
        "sample_count": collected["sample_count"],
        "scored_codes": collected.get("scored_codes", 0),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "patterns": results,
    })
    if not collected["sample_count"]:
        payload["notes"] = ["无可回测样本：先确保 sw3_member.is_leader 有龙头、且历史日线足够长。"]
    write_payload(PATTERNS_RESULT_FILE, payload)
    _print_patterns_summary(payload)
    return payload


def _print_patterns_summary(payload: Dict[str, Any]) -> None:
    print("=" * 108)
    print("  主力资金雷达 · 游资形态预测力后验 (patterns)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 候选龙头: {payload['candidate_count']}"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {display_path(PATTERNS_RESULT_FILE)}")
    print("-" * 108)
    results = payload.get("patterns", [])
    if not results:
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 108)
        return
    mid = str(VERIFY_HORIZONS[len(VERIFY_HORIZONS) // 2])
    print("  逐形态 × 逐周期超额收益(剔beta)；带 * = 方向正确、HAC |t|≥1.5 且通过 BH-FDR 10%")
    print(f"  {_ljust('形态', 6)}{_ljust('名称', 17)}{_ljust('阶段', 5)}{_ljust('信号', 5)}{_rjust('命中', 6)}  "
          + "".join(_rjust(f"{h}日", 9) for h in VERIFY_HORIZONS) + _rjust('有效@', 12))
    rows = sorted(results, key=lambda r: (r["horizons"][mid].get("excess_mean") or 0), reverse=True)
    for r in rows:
        hz = r["horizons"]
        cells = ""
        eff: List[str] = []
        for h in VERIFY_HORIZONS:
            d = hz[str(h)]
            ok = bool(d.get("fdr_10")) and _horizon_effective(r["signal"], d)
            cells += f"{(_pct(d.get('excess_mean')) + ('*' if ok else '')):>9}"
            if ok:
                eff.append(str(h))
        eff_str = (",".join(eff) + "日") if eff else "—"
        print(f"  {r['code']:<6}{_ljust(r['name'], 17)}{_ljust(r['phase'], 5)}{r['signal']:<5}"
              f"{hz[mid].get('n_hits', 0):>6}  {cells}{_rjust(eff_str, 12)}")
    print("-" * 108)
    print("  有效@ = buy 正超额、sell 负超额、hold 显著；observe只统计不升级；其余均要求 HAC |t|≥1.5 且通过 BH-FDR 10%。")
    print("=" * 108)


# ── distribution：出货分组合权重实验 ──────────────────────────

def _load_lhb_dates_by_code(conn: sqlite3.Connection) -> Tuple[Dict[str, List[str]], bool]:
    if not _table_exists(conn, LHB_TABLE):
        return {}, False
    out: Dict[str, List[str]] = {}
    rows = conn.execute(
        f"SELECT code, date FROM {LHB_TABLE} WHERE date IS NOT NULL ORDER BY code, date"
    ).fetchall()
    for row in rows:
        code = stock_storage._normalize_code(row["code"])
        if code:
            out.setdefault(code, []).append(str(row["date"]))
    return out, True


def _has_recent_lhb(lhb_dates: Dict[str, List[str]], code: str, as_of: str, cutoff: str) -> bool:
    dates = lhb_dates.get(code)
    if not dates:
        return False
    for d in reversed(dates):
        if d > as_of:
            continue
        return d > cutoff
    return False


def _collect_distribution_samples(conn: sqlite3.Connection,
                                  candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PIT 收集出货分候选特征 + 未来超额收益。"""
    max_h = max(DIST_EXPERIMENT_HORIZONS)
    series: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, int]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})

    if not series:
        return {"samples": [], "dates": [], "codes": [], "lhb_available": False}

    lhb_dates, lhb_available = _load_lhb_dates_by_code(conn)
    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    as_of_dates = _verify_as_of_dates(all_dates[:-max_h])
    cutoffs = {
        d: (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=CAPITAL_EVENT_DAYS)).strftime("%Y-%m-%d")
        for d in as_of_dates
    }

    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    for d in as_of_dates:
        date_rows: List[Dict[str, Any]] = []
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK - 1 or i + max_h >= len(bars):
                continue
            window = bars[i - LOOKBACK + 1:i + 1]
            res = _score_bars(code, window)
            if res is None:
                continue
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in DIST_EXPERIMENT_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            fired = match_patterns(code, window)
            pattern_codes = [p["code"] for p in fired]
            features = _distribution_model_features(
                res.get("signals", {}).get("technical_distribution_score", res.get("distribution_score")),
                pattern_codes,
                _has_recent_lhb(lhb_dates, code, d, cutoffs[d]),
                (res.get("sub_scores") or {}).get("divergence"),
            )
            date_rows.append({
                "date": d,
                "code": code,
                "features": features,
                "patterns": pattern_codes,
                "rets": rets,
            })

        if not date_rows:
            continue
        section_means = {
            h: _mean([row["rets"][h] for row in date_rows])
            for h in DIST_EXPERIMENT_HORIZONS
        }
        for row in date_rows:
            row["excess"] = {
                h: row["rets"][h] - section_means[h]
                for h in DIST_EXPERIMENT_HORIZONS
                if section_means[h] is not None
            }
        samples.extend(date_rows)
        used_dates.add(d)

    return {
        "samples": samples,
        "dates": sorted(used_dates),
        "codes": list(series.keys()),
        "lhb_available": lhb_available,
    }


def _rank_array(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _prepare_distribution_groups(samples: List[Dict[str, Any]],
                                 dates: Sequence[str]) -> Dict[int, List[Dict[str, Any]]]:
    date_set = set(dates)
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        if sample["date"] in date_set:
            by_date.setdefault(sample["date"], []).append(sample)

    groups: Dict[int, List[Dict[str, Any]]] = {h: [] for h in DIST_EXPERIMENT_HORIZONS}
    for rows in by_date.values():
        if len(rows) < VERIFY_MIN_NAMES:
            continue
        for h in DIST_EXPERIMENT_HORIZONS:
            valid = [row for row in rows if h in row.get("excess", {})]
            if len(valid) < VERIFY_MIN_NAMES:
                continue
            y = np.array([float(row["excess"][h]) for row in valid], dtype=float)
            yr = _rank_array(y)
            yc = yr - float(yr.mean())
            x = np.array(
                [[float(row["features"].get(feature, 0.0)) for feature in DIST_FEATURES] for row in valid],
                dtype=float,
            )
            groups[h].append({
                "x": x,
                "y": y,
                "y_rank_centered": yc,
                "y_rank_den": float(np.dot(yc, yc)),
            })
    return groups


def _round_metric(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


def _evaluate_distribution_weights(groups: Dict[int, List[Dict[str, Any]]],
                                   weights: Dict[str, float]) -> Dict[str, Any]:
    w = np.array([float(weights.get(feature, 0.0)) for feature in DIST_FEATURES], dtype=float)
    all_ics: List[float] = []
    all_spreads: List[float] = []
    by_horizon: Dict[str, Any] = {}

    for h in DIST_EXPERIMENT_HORIZONS:
        ics: List[float] = []
        spreads: List[float] = []
        for group in groups.get(h, []):
            scores = group["x"].dot(w)
            sr = _rank_array(scores)
            sc = sr - float(sr.mean())
            score_den = float(np.dot(sc, sc))
            target_den = group["y_rank_den"]
            if score_den > 0.0 and target_den > 0.0:
                ics.append(float(np.dot(sc, group["y_rank_centered"]) / math.sqrt(score_den * target_den)))
            order = np.argsort(scores, kind="mergesort")
            k = max(1, len(order) // VERIFY_BUCKETS)
            spreads.append(float(group["y"][order[-k:]].mean() - group["y"][order[:k]].mean()))
        all_ics.extend(ics)
        all_spreads.extend(spreads)
        by_horizon[str(h)] = {
            "ic_mean": _round_metric(_mean(ics)),
            "top_bottom_spread": _round_metric(_mean(spreads)),
            "n_sections": len(spreads),
        }

    ic_mean = _mean(all_ics)
    spread = _mean(all_spreads)
    objective = None
    if ic_mean is not None:
        objective = -ic_mean - (spread or 0.0) * DIST_EXPERIMENT_SPREAD_W
    return {
        "ic_mean": _round_metric(ic_mean),
        "top_bottom_spread": _round_metric(spread),
        "objective": _round_metric(objective, 5),
        "n_sections": len(all_spreads),
        "horizons": by_horizon,
    }


def _distribution_weight_grid(step: float = DIST_EXPERIMENT_GRID_STEP,
                              min_weight: float = DIST_EXPERIMENT_MIN_WEIGHT) -> List[Dict[str, float]]:
    """P14/P15/P20/P22固定5%，其余七项以10%为下限搜索，且包含生产权重。"""
    units = int(round(1.0 / step))
    fixed_weights = {
        feature: 0.05
        for feature in ("p14", "p15", "p20", "p22")
        if feature in DIST_FEATURES
    }
    fixed_units = {
        feature: int(round(weight / step)) for feature, weight in fixed_weights.items()
    }
    variable_features = [
        feature for feature in DIST_FEATURES if feature not in fixed_weights
    ]
    min_units = max(0, int(round(min_weight / step)))
    variable_units = units - sum(fixed_units.values())
    if min_units * len(variable_features) > variable_units:
        min_units = 0
    parts = [min_units] * len(variable_features)
    remaining_units = variable_units - min_units * len(variable_features)
    out: List[Dict[str, float]] = []

    def fill(idx: int, remaining: int) -> None:
        if idx == len(variable_features) - 1:
            parts[idx] = min_units + remaining
            weights = {
                feature: fixed_units[feature] / units for feature in fixed_units
            }
            weights.update({
                feature: parts[i] / units
                for i, feature in enumerate(variable_features)
            })
            out.append({feature: weights[feature] for feature in DIST_FEATURES})
            return
        for value in range(remaining + 1):
            parts[idx] = min_units + value
            fill(idx + 1, remaining - value)

    fill(0, remaining_units)
    return out


def _round_weights(weights: Dict[str, float]) -> Dict[str, float]:
    return {feature: round(float(weights.get(feature, 0.0)), 4) for feature in DIST_FEATURES}


def _distribution_model_report(name: str,
                               weights: Dict[str, float],
                               train_groups: Dict[int, List[Dict[str, Any]]],
                               val_groups: Dict[int, List[Dict[str, Any]]],
                               all_groups: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        "name": name,
        "weights": _round_weights(weights),
        "train": _evaluate_distribution_weights(train_groups, weights),
        "validation": _evaluate_distribution_weights(val_groups, weights),
        "all": _evaluate_distribution_weights(all_groups, weights),
    }


def _distribution_date_split(dates: Sequence[str]) -> Tuple[List[str], List[str]]:
    if len(dates) <= 1:
        return list(dates), []
    split = max(1, min(len(dates) - 1, int(round(len(dates) * 0.6))))
    return list(dates[:split]), list(dates[split:])


def _choose_distribution_model(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    eligible = []
    for row in rows:
        val = row.get("validation") or {}
        if (val.get("objective") or 0.0) <= 0.0:
            continue
        if (val.get("top_bottom_spread") or 0.0) > 0.0:
            continue
        eligible.append(row)
    if not eligible:
        eligible = [row for row in rows if (row.get("validation", {}).get("objective") or 0.0) > 0.0]
    if not eligible:
        eligible = rows
    return max(
        eligible,
        key=lambda row: (
            row.get("validation", {}).get("objective") or float("-inf"),
            row.get("train", {}).get("objective") or float("-inf"),
        ),
    ) if eligible else None


def run_distribution_experiment(max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_distribution_samples(conn, candidates)
    finally:
        conn.close()

    samples = collected["samples"]
    dates = collected["dates"]
    train_dates, val_dates = _distribution_date_split(dates)
    train_groups = _prepare_distribution_groups(samples, train_dates)
    val_groups = _prepare_distribution_groups(samples, val_dates)
    all_groups = _prepare_distribution_groups(samples, dates)

    payload = base_payload("distribution", len(candidates), pool=pool)
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "出货分权重实验：PIT 组合 P14/P15/P16/P17/P19/P20/P22/P26、近期龙虎榜、technical 和原始 divergence，检验未来超额收益是否随出货分升高而降低。",
        "params": {
            "features": list(DIST_FEATURES),
            "selected_weights": DIST_MODEL_WEIGHTS,
            "horizons": list(DIST_EXPERIMENT_HORIZONS),
            "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS,
            "min_names_per_section": VERIFY_MIN_NAMES,
            "grid_step": DIST_EXPERIMENT_GRID_STEP,
            "min_weight": DIST_EXPERIMENT_MIN_WEIGHT,
            "objective": "-RankIC - (高分组-低分组超额收益) * 3；越大表示出货风险排序越强",
        },
        "section_count": len(dates),
        "sample_count": len(samples),
        "scored_codes": len(collected["codes"]),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "date_split": {
            "train": [train_dates[0], train_dates[-1]] if train_dates else None,
            "validation": [val_dates[0], val_dates[-1]] if val_dates else None,
        },
        "lhb_available": collected["lhb_available"],
    })

    if not samples:
        payload["notes"] = ["无可实验样本：先确保 sw3_member.is_leader 有龙头、stock_history 足够长。"]
        write_payload(DIST_EXPERIMENT_FILE, payload)
        _print_distribution_summary(payload)
        return payload

    ablation_defs = [
        ("technical_only", {"technical": 1.0}),
        ("p14_only", {"p14": 1.0}),
        ("p15_only", {"p15": 1.0}),
        ("p16_only", {"p16": 1.0}),
        ("p17_only", {"p17": 1.0}),
        ("p19_only", {"p19": 1.0}),
        ("p20_only", {"p20": 1.0}),
        ("p22_only", {"p22": 1.0}),
        ("p26_only", {"p26": 1.0}),
        ("lhb_recent_only", {"lhb_recent": 1.0}),
        ("divergence_only", {"divergence": 1.0}),
        ("patterns_equal", {feature: 1 / 8 for feature in ("p14", "p15", "p16", "p17", "p19", "p20", "p22", "p26")}),
        ("all_equal", {feature: 1 / len(DIST_FEATURES) for feature in DIST_FEATURES}),
        ("selected", DIST_MODEL_WEIGHTS),
    ]

    train_ranked = []
    train_cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    for weights in _distribution_weight_grid():
        train_metrics = _evaluate_distribution_weights(train_groups, weights)
        objective = train_metrics.get("objective")
        if objective is None:
            continue
        key = tuple(_round_weights(weights)[feature] for feature in DIST_FEATURES)
        train_cache[key] = train_metrics
        train_ranked.append((objective, weights, train_metrics))
    train_ranked.sort(key=lambda item: item[0], reverse=True)

    shortlist: Dict[Tuple[float, ...], Dict[str, float]] = {}
    for _, weights, _ in train_ranked[:EXPERIMENT_VALIDATION_CANDIDATES]:
        rounded = _round_weights(weights)
        shortlist[tuple(rounded[feature] for feature in DIST_FEATURES)] = rounded
    candidates_grid = []
    for key, weights in shortlist.items():
        train_metrics = train_cache.get(key) or _evaluate_distribution_weights(train_groups, weights)
        candidates_grid.append({
            "weights": _round_weights(weights),
            "train": train_metrics,
            "validation": _evaluate_distribution_weights(val_groups, weights),
            "all": _evaluate_distribution_weights(all_groups, weights),
        })
    candidates_grid.sort(key=lambda row: row["train"].get("objective") or float("-inf"), reverse=True)
    top_models = candidates_grid[:10]
    validation_top = sorted(
        candidates_grid,
        key=lambda row: row["validation"].get("objective") or float("-inf"),
        reverse=True,
    )[:10]
    selected_report = _distribution_model_report(
        "selected", DIST_MODEL_WEIGHTS, train_groups, val_groups, all_groups
    )

    ablations = [
        _distribution_model_report(name, weights, train_groups, val_groups, all_groups)
        for name, weights in ablation_defs
    ]

    payload.update({
        "best_weights": top_models[0]["weights"] if top_models else None,
        "recommended_weights": selected_report.get("weights"),
        "selection_rule": "线上出货分使用人工指定权重；十一项特征均保留，单因子/分组消融仅作为报告参考。",
        "validation_candidate_count": len(candidates_grid),
        "top_models": top_models,
        "validation_top_models": validation_top,
        "ablations": ablations,
        "selected_model": selected_report,
        "recommended_model": selected_report,
        "notes": [
            "高分组-低分组超额收益为负时，说明出货分高的一组未来相对收益更差。",
            "divergence 使用原始 div 分，不使用吸筹侧反向后的 div_eff。",
        ],
    })
    write_payload(DIST_EXPERIMENT_FILE, payload)
    _print_distribution_summary(payload)
    return payload


def _print_distribution_summary(payload: Dict[str, Any]) -> None:
    print("=" * 108)
    print("  主力资金雷达 · 出货分权重实验 (distribution)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 候选龙头: {payload['candidate_count']}"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {display_path(DIST_EXPERIMENT_FILE)}")
    print("-" * 108)
    if payload.get("status") != "ok":
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 108)
        return

    def metric_text(metrics: Dict[str, Any]) -> str:
        return (f"IC {_fmt(metrics.get('ic_mean')):>7} · "
                f"高低差 {_pct(metrics.get('top_bottom_spread')).strip():>7} · "
                f"obj {_fmt(metrics.get('objective')):>7}")

    print("  Top 网格模型（train 排序；validation 用时间后段验证）:")
    for i, row in enumerate(payload.get("top_models", [])[:5], 1):
        print(f"  {i:>2}. w={row['weights']} | train {metric_text(row['train'])} | val {metric_text(row['validation'])}")
    print("-" * 108)
    print("  Top 网格模型（validation 排序）:")
    for i, row in enumerate(payload.get("validation_top_models", [])[:5], 1):
        print(f"  {i:>2}. w={row['weights']} | train {metric_text(row['train'])} | val {metric_text(row['validation'])}")
    print("-" * 108)
    print("  消融实验:")
    for row in payload.get("ablations", []):
        print(f"  {_ljust(row['name'], 22)} train {metric_text(row['train'])} | val {metric_text(row['validation'])}")
    print("-" * 108)
    selected = payload.get("selected_model", {})
    print(f"  当前线上出货分: w={selected.get('weights')} | all {metric_text(selected.get('all', {}))}")
    rec = payload.get("recommended_model") or {}
    print(f"  推荐出货分: w={rec.get('weights')} | val {metric_text(rec.get('validation', {}))}")
    print("=" * 108)


# ── accumulation：十三原始分吸筹总分权重实验 ────────────────────

def _load_capital_histories(conn: sqlite3.Connection) -> Dict[str, Any]:
    holder_dates: Dict[str, List[str]] = {}
    holder_values: Dict[str, List[float]] = {}
    repurchase_dates: Dict[str, List[str]] = {}
    if _table_exists(conn, HOLDER_TABLE):
        rows = conn.execute(
            f"SELECT code, disclose_date, change_pct FROM {HOLDER_TABLE} "
            "WHERE disclose_date IS NOT NULL AND change_pct IS NOT NULL ORDER BY code, disclose_date"
        ).fetchall()
        for row in rows:
            code = stock_storage._normalize_code(row["code"])
            if not code:
                continue
            holder_dates.setdefault(code, []).append(str(row["disclose_date"]))
            holder_values.setdefault(code, []).append(float(row["change_pct"]))
    if _table_exists(conn, REPURCHASE_TABLE):
        rows = conn.execute(
            f"SELECT code, disclose_date FROM {REPURCHASE_TABLE} "
            "WHERE disclose_date IS NOT NULL ORDER BY code, disclose_date"
        ).fetchall()
        for row in rows:
            code = stock_storage._normalize_code(row["code"])
            if code:
                repurchase_dates.setdefault(code, []).append(str(row["disclose_date"]))
    return {
        "holder_dates": holder_dates,
        "holder_values": holder_values,
        "repurchase_dates": repurchase_dates,
        "holder_available": bool(holder_dates),
        "repurchase_available": bool(repurchase_dates),
    }


def _recent_date_hit(dates: Dict[str, List[str]], code: str, as_of: str, cutoff: str) -> bool:
    ds = dates.get(code)
    if not ds:
        return False
    idx = bisect.bisect_right(ds, as_of) - 1
    return idx >= 0 and ds[idx] > cutoff


def _holder_change_at(histories: Dict[str, Any], code: str, as_of: str) -> Optional[float]:
    holder_dates = histories["holder_dates"].get(code)
    if not holder_dates:
        return None
    idx = bisect.bisect_right(holder_dates, as_of) - 1
    if idx < 0:
        return None
    return float(histories["holder_values"][code][idx])


def _accumulation_capital_at(
    histories: Dict[str, Any], code: str, as_of: str,
) -> Dict[str, Any]:
    """返回单日可见的吸筹资金面特征，供历史评分统一做 PIT 截断。"""
    if not histories or not as_of:
        return {"holder_change": None, "repurchase_recent": False}
    cutoff = (
        datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=CAPITAL_EVENT_DAYS)
    ).strftime("%Y-%m-%d")
    return {
        "holder_change": _holder_change_at(histories, code, as_of),
        "repurchase_recent": _recent_date_hit(
            histories.get("repurchase_dates") or {}, code, as_of, cutoff,
        ),
    }


def _collect_accumulation_samples(
    conn: sqlite3.Connection,
    candidates: List[Dict[str, Any]],
    pool: str = DEFAULT_POOL,
) -> Dict[str, Any]:
    """PIT 收集十三个吸筹原始特征 + 未来超额收益。"""
    max_h = max(ACCUM_EXPERIMENT_HORIZONS)
    series: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, int]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})
    if not series:
        return {"samples": [], "dates": [], "codes": [], "holder_available": False, "repurchase_available": False}

    histories = _load_capital_histories(conn)
    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    as_of_dates = _verify_as_of_dates(all_dates[:-max_h])
    p1_market_gates = _p1_market_gate_by_date(conn, as_of_dates)
    cutoffs = {
        d: (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=CAPITAL_EVENT_DAYS)).strftime("%Y-%m-%d")
        for d in as_of_dates
    }

    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    for d in as_of_dates:
        date_rows: List[Dict[str, Any]] = []
        cutoff = cutoffs[d]
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK - 1 or i + max_h >= len(bars):
                continue
            window = bars[i - LOOKBACK + 1:i + 1]
            res = _score_bars(code, window)
            if res is None:
                continue
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in ACCUM_EXPERIMENT_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            fired = match_patterns(
                code, window, pool=pool,
                p1_market_gate=p1_market_gates.get(d, False),
            )
            pattern_codes = [p["code"] for p in fired]
            feature_row = {
                "sub_scores": res.get("sub_scores") or {},
                "patterns": pattern_codes,
                "holder_change": _holder_change_at(histories, code, d),
                "repurchase_recent": _recent_date_hit(histories["repurchase_dates"], code, d, cutoff),
            }
            date_rows.append({
                "date": d,
                "code": code,
                "features": _accumulation_raw_features(feature_row),
                "rets": rets,
            })

        if not date_rows:
            continue
        section_means = {
            h: _mean([row["rets"][h] for row in date_rows])
            for h in ACCUM_EXPERIMENT_HORIZONS
        }
        for row in date_rows:
            row["excess"] = {
                h: row["rets"][h] - section_means[h]
                for h in ACCUM_EXPERIMENT_HORIZONS
                if section_means[h] is not None
            }
        samples.extend(date_rows)
        used_dates.add(d)

    return {
        "samples": samples,
        "dates": sorted(used_dates),
        "codes": list(series.keys()),
        "holder_available": histories["holder_available"],
        "repurchase_available": histories["repurchase_available"],
    }


def _prepare_accumulation_groups(samples: List[Dict[str, Any]],
                                 dates: Sequence[str]) -> Dict[int, List[Dict[str, Any]]]:
    date_set = set(dates)
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        if sample["date"] in date_set:
            by_date.setdefault(sample["date"], []).append(sample)

    groups: Dict[int, List[Dict[str, Any]]] = {h: [] for h in ACCUM_EXPERIMENT_HORIZONS}
    for rows in by_date.values():
        if len(rows) < VERIFY_MIN_NAMES:
            continue
        for h in ACCUM_EXPERIMENT_HORIZONS:
            valid = [row for row in rows if h in row.get("excess", {})]
            if len(valid) < VERIFY_MIN_NAMES:
                continue
            y = np.array([float(row["excess"][h]) for row in valid], dtype=float)
            yr = _rank_array(y)
            yc = yr - float(yr.mean())
            x = np.array(
                [[float(row["features"].get(feature, 0.0)) for feature in ACCUM_FEATURES] for row in valid],
                dtype=float,
            )
            groups[h].append({
                "x": x,
                "y": y,
                "y_rank_centered": yc,
                "y_rank_den": float(np.dot(yc, yc)),
            })
    return groups


def _evaluate_accumulation_weights(groups: Dict[int, List[Dict[str, Any]]],
                                   weights: Dict[str, float]) -> Dict[str, Any]:
    w = np.array([float(weights.get(feature, 0.0)) for feature in ACCUM_FEATURES], dtype=float)
    all_ics: List[float] = []
    all_spreads: List[float] = []
    by_horizon: Dict[str, Any] = {}

    for h in ACCUM_EXPERIMENT_HORIZONS:
        ics: List[float] = []
        spreads: List[float] = []
        for group in groups.get(h, []):
            scores = group["x"].dot(w)
            sr = _rank_array(scores)
            sc = sr - float(sr.mean())
            score_den = float(np.dot(sc, sc))
            target_den = group["y_rank_den"]
            if score_den > 0.0 and target_den > 0.0:
                ics.append(float(np.dot(sc, group["y_rank_centered"]) / math.sqrt(score_den * target_den)))
            order = np.argsort(scores, kind="mergesort")
            k = max(1, len(order) // VERIFY_BUCKETS)
            spreads.append(float(group["y"][order[-k:]].mean() - group["y"][order[:k]].mean()))
        all_ics.extend(ics)
        all_spreads.extend(spreads)
        ic_mean = _mean(ics)
        spread = _mean(spreads)
        by_horizon[str(h)] = {
            "ic_mean": _round_metric(ic_mean),
            "top_bottom_spread": _round_metric(spread),
            "objective": _round_metric((ic_mean or 0.0) + (spread or 0.0) * ACCUM_EXPERIMENT_SPREAD_W, 5) if ics else None,
            "n_sections": len(spreads),
        }

    ic_mean = _mean(all_ics)
    spread = _mean(all_spreads)
    objective = None if ic_mean is None else ic_mean + (spread or 0.0) * ACCUM_EXPERIMENT_SPREAD_W
    return {
        "ic_mean": _round_metric(ic_mean),
        "top_bottom_spread": _round_metric(spread),
        "objective": _round_metric(objective, 5),
        "n_sections": len(all_spreads),
        "horizons": by_horizon,
    }


def _accumulation_weight_grid(step: float = ACCUM_EXPERIMENT_GRID_STEP,
                              min_weight: float = ACCUM_EXPERIMENT_MIN_WEIGHT) -> List[Dict[str, float]]:
    """权重网格：生产5%因子固定，其余七项在剩余70%内搜索。"""
    units = int(round(1.0 / step))
    fixed_weights = {
        feature: 0.05
        for feature in ("p2", "p5", "p23", "p24", "p25", "repurchase")
        if feature in ACCUM_FEATURES
    }
    fixed_units = {feature: int(round(weight / step)) for feature, weight in fixed_weights.items()}
    variable_features = [feature for feature in ACCUM_FEATURES if feature not in fixed_weights]
    variable_units = units - sum(fixed_units.values())
    min_units = max(0, int(round(min_weight / step)))
    if min_units * len(variable_features) > variable_units:
        min_units = 0
    parts = [min_units] * len(variable_features)
    remaining_units = variable_units - min_units * len(variable_features)
    out: List[Dict[str, float]] = []

    def fill(idx: int, remaining: int) -> None:
        if idx == len(variable_features) - 1:
            parts[idx] = min_units + remaining
            weights = {feature: fixed_units[feature] / units for feature in fixed_units}
            weights.update({feature: parts[i] / units for i, feature in enumerate(variable_features)})
            out.append({feature: weights[feature] for feature in ACCUM_FEATURES})
            return
        for value in range(remaining + 1):
            parts[idx] = min_units + value
            fill(idx + 1, remaining - value)

    fill(0, remaining_units)
    return out


def _round_accum_weights(weights: Dict[str, float]) -> Dict[str, float]:
    return {feature: round(float(weights.get(feature, 0.0)), 4) for feature in ACCUM_FEATURES}


def _accumulation_model_report(name: str,
                               weights: Dict[str, float],
                               train_groups: Dict[int, List[Dict[str, Any]]],
                               val_groups: Dict[int, List[Dict[str, Any]]],
                               all_groups: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        "name": name,
        "weights": _round_accum_weights(weights),
        "train": _evaluate_accumulation_weights(train_groups, weights),
        "validation": _evaluate_accumulation_weights(val_groups, weights),
        "all": _evaluate_accumulation_weights(all_groups, weights),
    }


def _choose_accumulation_model(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    eligible = []
    for row in rows:
        val = row.get("validation") or {}
        if (val.get("objective") or 0.0) <= 0.0:
            continue
        if (val.get("top_bottom_spread") or 0.0) < 0.0:
            continue
        eligible.append(row)
    if not eligible:
        eligible = [
            row for row in rows
            if (row.get("validation", {}).get("objective") or 0.0) > 0.0
            and (row.get("validation", {}).get("top_bottom_spread") or 0.0) >= 0.0
        ]
    if not eligible:
        eligible = rows
    return max(
        eligible,
        key=lambda row: (
            row.get("validation", {}).get("objective") or float("-inf"),
            row.get("train", {}).get("objective") or float("-inf"),
        ),
    ) if eligible else None


def run_accumulation_experiment(max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_accumulation_samples(conn, candidates, pool=pool)
    finally:
        conn.close()

    samples = collected["samples"]
    dates = collected["dates"]
    train_dates, val_dates = _distribution_date_split(dates)
    train_groups = _prepare_accumulation_groups(samples, train_dates)
    val_groups = _prepare_accumulation_groups(samples, val_dates)
    all_groups = _prepare_accumulation_groups(samples, dates)

    payload = base_payload("accumulation", len(candidates), pool=pool)
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "吸筹总分权重实验：用 chip、position、cmf_eff、P1、P2、P3、P5、P21、P23、P24、P25、股东户数变化、公司回购十三个原始分直接加权，不用截面 rank 合成线上分数。P23/P25按核心有效口径参与计分。",
        "params": {
            "horizons": list(ACCUM_EXPERIMENT_HORIZONS),
            "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS,
            "min_names_per_section": VERIFY_MIN_NAMES,
            "grid_step": ACCUM_EXPERIMENT_GRID_STEP,
            "min_weight": ACCUM_EXPERIMENT_MIN_WEIGHT,
            "min_weight_scope": "除固定5%因子外的七项",
            "fixed_weights": {
                "p2": 0.05, "p5": 0.05, "p23": 0.05,
                "p24": 0.05, "p25": 0.05, "repurchase": 0.05,
            },
            "features": list(ACCUM_FEATURES),
            "score_formula": "sum(weight_i * raw_feature_i)，raw_feature_i 均为 0~100 原始分",
            "objective": "RankIC + (高分组-低分组超额收益) * 3；rank 只用于评价，不进入线上合成",
            "selected_weights": ACCUM_MODEL_WEIGHTS,
            "missing_holder_change_score": 50.0,
        },
        "section_count": len(dates),
        "sample_count": len(samples),
        "scored_codes": len(collected["codes"]),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "date_split": {
            "train": [train_dates[0], train_dates[-1]] if train_dates else None,
            "validation": [val_dates[0], val_dates[-1]] if val_dates else None,
        },
        "holder_available": collected["holder_available"],
        "repurchase_available": collected["repurchase_available"],
    })
    if not samples:
        payload["notes"] = ["无可实验样本：先确保 sw3_member.is_leader 有龙头、stock_history 和资金面数据足够长。"]
        write_payload(ACCUM_EXPERIMENT_FILE, payload)
        _print_accumulation_summary(payload)
        return payload

    def one_hot(feature: str) -> Dict[str, float]:
        return {k: (1.0 if k == feature else 0.0) for k in ACCUM_FEATURES}

    market_structure_features = (
        "chip", "position", "cmf_eff", "p1", "p2", "p3", "p5",
        "p21", "p23", "p24", "p25",
    )
    market_structure_equal = {
        k: (1 / len(market_structure_features) if k in market_structure_features else 0.0)
        for k in ACCUM_FEATURES
    }
    capital_equal = {
        k: (1 / 2 if k in ("holder_change", "repurchase") else 0.0)
        for k in ACCUM_FEATURES
    }
    equal = {k: 1.0 / len(ACCUM_FEATURES) for k in ACCUM_FEATURES}
    ablation_defs = [
        *[(f"{feature}_only", one_hot(feature)) for feature in ACCUM_FEATURES],
        ("market_structure_equal", market_structure_equal),
        ("capital_equal", capital_equal),
        ("all_equal", equal),
        ("selected", ACCUM_MODEL_WEIGHTS),
    ]

    train_ranked = []
    train_cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    for weights in _accumulation_weight_grid():
        train_metrics = _evaluate_accumulation_weights(train_groups, weights)
        if train_metrics.get("objective") is None:
            continue
        key = tuple(_round_accum_weights(weights)[feature] for feature in ACCUM_FEATURES)
        train_cache[key] = train_metrics
        train_ranked.append((train_metrics.get("objective") or float("-inf"), weights, train_metrics))
    train_ranked.sort(key=lambda item: item[0], reverse=True)

    shortlist: Dict[Tuple[float, ...], Dict[str, float]] = {}
    for _, weights, _ in train_ranked[:EXPERIMENT_VALIDATION_CANDIDATES]:
        rounded = _round_accum_weights(weights)
        shortlist[tuple(rounded[feature] for feature in ACCUM_FEATURES)] = rounded
    candidates_grid = []
    for key, weights in shortlist.items():
        train_metrics = train_cache.get(key) or _evaluate_accumulation_weights(train_groups, weights)
        candidates_grid.append({
            "weights": _round_accum_weights(weights),
            "train": train_metrics,
            "validation": _evaluate_accumulation_weights(val_groups, weights),
            "all": _evaluate_accumulation_weights(all_groups, weights),
        })
    candidates_grid.sort(key=lambda row: row["train"].get("objective") or float("-inf"), reverse=True)
    top_models = candidates_grid[:10]
    validation_top = sorted(
        candidates_grid,
        key=lambda row: row["validation"].get("objective") or float("-inf"),
        reverse=True,
    )[:10]
    recommended = _choose_accumulation_model(candidates_grid)
    recommended_weights = (recommended or {}).get("weights") or ACCUM_MODEL_WEIGHTS
    ablations = [
        *[
            _accumulation_model_report(name, weights, train_groups, val_groups, all_groups)
            for name, weights in ablation_defs
        ],
        _accumulation_model_report("recommended", recommended_weights, train_groups, val_groups, all_groups),
    ]

    payload.update({
        "accumulation": {
            "best_train_weights": top_models[0]["weights"] if top_models else None,
            "recommended_weights": _round_accum_weights(recommended_weights),
            "selection_rule": "P2/P5/P23/P24/P25/回购固定各5%，其余七项在剩余70%中按5%步长且各自不低于5%搜索；单因子/分组消融仅作为报告参考，不参与推荐。",
            "validation_candidate_count": len(candidates_grid),
            "top_models": top_models,
            "validation_top_models": validation_top,
            "ablations": ablations,
            "selected_model": ablations[-2],
            "recommended_model": recommended,
        },
        "notes": [
            "线上吸筹总分使用 chip、position、cmf_eff、P1、P2、P3、P5、P21、P23、P24、P25、股东户数变化、公司回购十三个原始分按权重相加；P23/P25按核心有效口径参与计分。",
            "实验里的 RankIC 仅用于评价分数排序能力；高低差是高分五分位相对低分五分位的未来超额收益差。",
        ],
    })
    write_payload(ACCUM_EXPERIMENT_FILE, payload)
    _print_accumulation_summary(payload)
    return payload


def _print_accumulation_summary(payload: Dict[str, Any]) -> None:
    print("=" * 112)
    print("  主力资金雷达 · 十三原始分吸筹总分权重实验 (accumulation)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 候选龙头: {payload['candidate_count']}"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {display_path(ACCUM_EXPERIMENT_FILE)}")
    print("-" * 112)
    if payload.get("status") != "ok":
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 112)
        return

    def metric_text(metrics: Dict[str, Any]) -> str:
        return (f"IC {_fmt(metrics.get('ic_mean')):>7} · "
                f"高低差 {_pct(metrics.get('top_bottom_spread')).strip():>7} · "
                f"obj {_fmt(metrics.get('objective')):>7}")

    acc = payload.get("accumulation", {})
    print("  Top 网格模型（train 排序；validation 时间后段验证）:")
    for i, row in enumerate(acc.get("top_models", [])[:5], 1):
        print(f"  {i:>2}. w={row['weights']} | train {metric_text(row['train'])} | val {metric_text(row['validation'])}")
    print("-" * 112)
    print("  Top 网格模型（validation 排序）:")
    for i, row in enumerate(acc.get("validation_top_models", [])[:5], 1):
        print(f"  {i:>2}. w={row['weights']} | train {metric_text(row['train'])} | val {metric_text(row['validation'])}")
    print("-" * 112)
    print("  消融实验（selected 为当前线上参数，recommended 为实验建议）:")
    for row in acc.get("ablations", []):
        print(f"  {_ljust(row['name'], 28)} train {metric_text(row['train'])} | val {metric_text(row['validation'])}")
    rec = acc.get("recommended_model") or {}
    print("-" * 112)
    print(f"  推荐参数: w={rec.get('weights')} | val {metric_text(rec.get('validation', {}))}")
    print("=" * 112)


# ── watch（骨架）──────────────────────────────────────────────

def run_watch(max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    """实时交易监控入口（骨架）：取候选龙头占位，后续接盘口/实时行情事件。"""
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
    finally:
        conn.close()
    payload = base_payload("watch", len(candidates), pool=pool)
    payload.update({
        "status": "scaffold",
        "description": "实时交易监控框架（待接入实时行情/盘口）。",
        "stocks": [{**c, "watch_status": "TODO", "last_trade_snapshot": None} for c in candidates],
    })
    write_payload(WATCH_STATE_FILE, payload)
    print(f"[watch] scaffold written · candidates={len(candidates)}")
    return payload


# ── latent：潜伏妖股观察名单（"太极 2022 式"左侧筛选）─────────────

def _yaogu_genes(conn: sqlite3.Connection, codes: Sequence[str],
                 as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """妖股基因：全史龙虎榜次数 + 近 N 年最大换手 + 涨停日数 → 池内分位等权 gene(0~1)。PIT 用 as_of 截断。"""
    end = as_of or datetime.now().strftime("%Y-%m-%d")
    start3y = (datetime.strptime(end, "%Y-%m-%d")
               - timedelta(days=int(LATENT_GENE_LOOKBACK_YEARS * 365))).strftime("%Y-%m-%d")
    raw: Dict[str, Dict[str, float]] = {}
    for c in codes:
        lhb = conn.execute("SELECT COUNT(*) FROM lhb_all WHERE code=? AND date<=?", (c, end)).fetchone()[0]
        r = conn.execute(
            "SELECT MAX(daily_turnover_rate), "
            "SUM(CASE WHEN daily_change_pct>=9.5 THEN 1 ELSE 0 END) "
            "FROM stock_history WHERE code=? AND date BETWEEN ? AND ?", (c, start3y, end)).fetchone()
        raw[c] = {"lhb_all": float(lhb or 0), "max_turn": float(r[0] or 0.0), "limitups_3y": float(r[1] or 0)}

    def pctile(key: str) -> Dict[str, float]:
        vals = sorted(v[key] for v in raw.values())
        n = len(vals) or 1
        return {c: bisect.bisect_right(vals, raw[c][key]) / n for c in raw}

    p_lhb, p_turn, p_lim = pctile("lhb_all"), pctile("max_turn"), pctile("limitups_3y")
    return {c: {**raw[c], "gene": (p_lhb[c] + p_turn[c] + p_lim[c]) / 3.0} for c in raw}


NEWS_CATALYST_DAYS = 30   # latent 催化剂窗口：近 N 日新闻


def _news_catalyst(conn: sqlite3.Connection, code: str, as_of: Optional[str] = None,
                   days: int = NEWS_CATALYST_DAYS) -> Optional[Dict[str, Any]]:
    """近 days 日新闻催化剂(stock_crawl_news.py 入库, 缺表则 None)：条数 + 最新日 + Top 题材标签。

    PIT：as_of 给定时只取该日及以前的新闻。⚠️新闻预测力未验证，仅作展示/证据，不进 latent 排序。
    """
    end_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
    since = (end_dt - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    until = (as_of + " 23:59:59") if as_of else None
    rows = stock_storage.load_recent_news(conn, code, since=since, until=until, limit=50)
    if not rows:
        return None
    themes: Counter = Counter()
    for r in rows:
        for t in (r.get("themes") or "").split(","):
            t = t.strip()
            if t:
                themes[t] += 1
    latest = str(rows[0].get("pub_time") or "")[:10]
    try:
        age = (end_dt - datetime.strptime(latest, "%Y-%m-%d")).days
    except ValueError:
        age = None
    return {"news_count": len(rows), "latest_date": latest, "latest_age_days": age,
            "themes": [t for t, _ in themes.most_common(3)]}


def run_latent(as_of: Optional[str] = None, max_cap: Optional[float] = None,
               pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    """潜伏妖股观察名单：复用 ambush 打分核心，按左侧硬过滤 + 综合潜伏分排名。

    硬过滤：位置<LATENT_MAX_POS ∩ 换手分位<LATENT_MAX_TURN_PCTILE ∩ 近90日未上龙虎榜
            ∩ 出货分<LATENT_MAX_DIST ∩ 未连板未触发突破。
    潜伏分：吸筹分位 + 妖股基因 + 低位 + 板块题材热 + (户数降/回购证据)。详见研究报告与纪要。
    """
    amb = run_ambush(as_of=as_of, max_cap=max_cap, pool=pool, write=False, print_summary=False)
    rows = amb.get("stocks", [])
    survivors: List[Dict[str, Any]] = []
    conn = stock_storage.connect(DB_FILE)
    try:
        genes = _yaogu_genes(conn, [r["code"] for r in rows], as_of=as_of)
        for r in rows:
            sig = r.get("signals", {})
            pos = sig.get("close_pctile")
            turn = sig.get("turnover_pctile")
            dist = r.get("distribution_score") or 0.0
            streak = sig.get("limit_streak") or 0
            if pos is None or pos > LATENT_MAX_POS:
                continue
            if turn is not None and turn > LATENT_MAX_TURN_PCTILE:
                continue
            if r.get("lhb_recent") or dist > LATENT_MAX_DIST:
                continue
            if streak >= 1 or r.get("triggered"):
                continue
            g = genes.get(r["code"], {"gene": 0.0, "lhb_all": 0, "max_turn": 0.0, "limitups_3y": 0})
            acc = (r.get("accumulation_percentile") or 0.0) / 100.0
            heat = (r.get("sw2_heat_pctile") or 0.0) / 100.0
            hc = r.get("holder_change")
            evid = ((LATENT_HOLDER_BONUS if (hc is not None and hc < 0) else 0.0)
                    + (LATENT_REPO_BONUS if r.get("repurchase_recent") else 0.0))
            latent = (LATENT_WEIGHTS["accumulation"] * acc + LATENT_WEIGHTS["gene"] * g["gene"]
                      + LATENT_WEIGHTS["low_pos"] * (1.0 - pos) + LATENT_WEIGHTS["theme_heat"] * heat + evid)
            r["yaogu_gene"] = round(g["gene"] * 100.0, 1)
            r["latent_score"] = round(latent * 100.0, 1)
            r["latent_signals"] = {"lhb_all": int(g["lhb_all"]), "max_turn_3y": round(g["max_turn"], 1),
                                   "limitups_3y": int(g["limitups_3y"])}
            cat = _news_catalyst(conn, r["code"], as_of=as_of)   # ①题材层(新闻):展示/证据,不进排序
            if cat:
                r["news_catalyst"] = cat
            survivors.append(r)
    finally:
        conn.close()
    survivors.sort(key=lambda x: x["latent_score"], reverse=True)

    payload = base_payload("latent", len(rows), pool=pool)
    payload.update({
        "status": "ok" if survivors else "empty",
        "description": "潜伏妖股观察名单：左侧低位+安静+吸筹+妖股基因+真吸筹证据。⚠️观察名单非买点触发器(左侧弱/慢)。",
        "pool": pool,
        "as_of": as_of,
        "market_regime": amb.get("market_regime"),
        "params": {
            "max_pos": LATENT_MAX_POS, "max_turn_pctile": LATENT_MAX_TURN_PCTILE,
            "max_distribution": LATENT_MAX_DIST, "weights": LATENT_WEIGHTS,
            "holder_bonus": LATENT_HOLDER_BONUS, "repurchase_bonus": LATENT_REPO_BONUS,
            "gene_lookback_years": LATENT_GENE_LOOKBACK_YEARS,
        },
        "screened_count": len(survivors),
        "pool_size": len(rows),
        "stocks": survivors,
    })
    write_payload(LATENT_RESULT_FILE, payload)
    _print_latent_summary(payload)
    return payload


def _print_latent_summary(payload: Dict[str, Any]) -> None:
    print("=" * 108)
    print("  主力资金雷达 · 潜伏妖股观察名单 (latent)")
    print(f"  生成时间: {payload['generated_at']} · 池: {payload.get('pool')}({payload.get('pool_size',0)}) "
          f"· 入选: {payload.get('screened_count',0)} · 落盘: {display_path(LATENT_RESULT_FILE)}")
    mr = payload.get("market_regime")
    if isinstance(mr, dict) and mr.get("available"):
        print(f"  大盘: {'站上' if mr.get('favorable') else '跌破'}MA20 — {mr.get('note','')}")
    print("  硬过滤: 位置<{:.0%} ∩ 换手分位<{:.0%} ∩ 近90日未上榜 ∩ 出货<{:.0f} ∩ 未启动 ；⚠️观察名单非买点".format(
        LATENT_MAX_POS, LATENT_MAX_TURN_PCTILE, LATENT_MAX_DIST))
    print("-" * 108)
    stocks = payload.get("stocks", [])
    if not stocks:
        print("  （无入选——可能池空或当前无左侧低位安静标的）")
        print("=" * 108)
        return
    header = (_ljust("代码", 8) + _ljust("名称", 11) + _rjust("潜伏", 6) + _rjust("位置", 6)
              + _rjust("换手", 6) + _rjust("吸筹%", 7) + _rjust("基因", 6) + _rjust("户数Δ", 8)
              + _rjust("回购", 5) + _rjust("板块热", 7) + "  行业/题材")
    print("  " + header)
    for s in stocks[:25]:
        sig = s.get("signals", {})
        pos = (sig.get("close_pctile") or 0) * 100
        turn = (sig.get("turnover_pctile") or 0) * 100
        hc = s.get("holder_change")
        hc_s = f"{hc:+.0f}%" if hc is not None else "-"
        heat = s.get("sw2_heat_pctile")
        heat_s = f"{heat:.0f}" if heat is not None else "-"
        cat = s.get("news_catalyst")
        if cat and cat.get("themes"):
            seg = f"📰{cat['news_count']} {'/'.join(cat['themes'][:2])}"   # 新闻题材优先
        elif cat:
            seg = f"📰{cat['news_count']} " + (s.get("tracking_theme") or s.get("segment_name") or "")
        else:
            seg = (s.get("tracking_theme") or s.get("segment_name") or "")
        row = (_ljust(s["code"], 8) + _ljust(s.get("name", "")[:5], 11)
               + _rjust(s.get("latent_score", 0), 6) + _rjust(f"{pos:.0f}%", 6) + _rjust(f"{turn:.0f}%", 6)
               + _rjust(f"{s.get('accumulation_percentile') or 0:.0f}", 7)
               + _rjust(f"{s.get('yaogu_gene') or 0:.0f}", 6) + _rjust(hc_s, 8)
               + _rjust("✓" if s.get("repurchase_recent") else "", 5) + _rjust(heat_s, 7) + "  " + seg[:16])
        print("  " + row)
    print("=" * 108)


# ── CLI ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="主力资金雷达")
    parser.add_argument(
        "mode", nargs="?", choices=MODES, default=DEFAULT_MODE,
        help="运行模式：ambush(默认,吸筹分+形态) / watch(实时监控) / "
             "verify(吸筹分回测) / patterns(形态预测力回测) / "
             "distribution(出货分权重实验) / accumulation(十三原始分吸筹总分权重实验) / "
             "latent(潜伏妖股观察名单:左侧低位+安静+吸筹+妖股基因, 建议配 --pool hotmoney)",
    )
    parser.add_argument(
        "--as-of", default=None, metavar="YYYY-MM-DD",
        help="ambush 模式只用该日期及以前的 bar（PIT 历史复盘，并禁用题材缓存防泄漏）；默认最新交易日。",
    )
    parser.add_argument(
        "--exclude-large-cap", action=argparse.BooleanOptionalAction, default=False,
        help=f"是否剔除大盘股（默认否=纳入全市值龙头）；用 --exclude-large-cap 只看 ≤{MAX_MARKET_CAP_YI:g}亿小/中盘做市值分层查看。",
    )
    parser.add_argument(
        "--pool", choices=POOLS, default=DEFAULT_POOL,
        help="候选池：leader=细分龙头(默认) / hotmoney=游资小盘 / etf=stock_etf_pool.py 配置池。",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="patterns 回测并行进程数；默认1，完整双池建议按CPU情况使用4~6。",
    )
    parser.add_argument(
        "--pattern-min", type=int, default=1, choices=range(1, 27),
        help="patterns 回测的最小形态编号；与 --pattern-max 组合可只测指定编号区间。",
    )
    parser.add_argument(
        "--pattern-max", type=int, default=26, choices=range(1, 27),
        help="patterns 回测的最大形态编号；例如24表示仅回测P1-P24。",
    )
    return parser


def run_mode(mode: str, as_of: Optional[str] = None,
             max_cap: Optional[float] = None, pool: str = DEFAULT_POOL,
             jobs: int = 1, pattern_min: int = 1,
             pattern_max: int = 26) -> Dict[str, Any]:
    if mode == "watch":
        return run_watch(max_cap=max_cap, pool=pool)
    if mode == "verify":
        return run_verify(max_cap=max_cap, pool=pool)
    if mode == "patterns":
        return run_patterns(
            max_cap=max_cap,
            pool=pool,
            jobs=jobs,
            pattern_min=pattern_min,
            pattern_max=pattern_max,
        )
    if mode == "distribution":
        return run_distribution_experiment(max_cap=max_cap, pool=pool)
    if mode == "accumulation":
        return run_accumulation_experiment(max_cap=max_cap, pool=pool)
    if mode == "latent":
        return run_latent(as_of=as_of, max_cap=max_cap, pool=pool)
    return run_ambush(as_of=as_of, max_cap=max_cap, pool=pool)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    # hotmoney 已在建池时过滤；ETF 无公司总市值语义。市值过滤仅对 leader 生效。
    max_cap = MAX_MARKET_CAP_YI if (args.exclude_large_cap and args.pool == "leader") else None
    return run_mode(
        args.mode,
        as_of=args.as_of,
        max_cap=max_cap,
        pool=args.pool,
        jobs=args.jobs,
        pattern_min=args.pattern_min,
        pattern_max=args.pattern_max,
    )


if __name__ == "__main__":
    main()
