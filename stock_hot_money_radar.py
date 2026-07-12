"""主力资金雷达。

候选池 = 细分行业龙头（stock_crawl_segment_leaders 选出、回写主库 sw3_member.is_leader）。
潜伏分目标：捕捉「左侧吸筹」——主力在低位悄悄建仓、但价格还没起飞的阶段。

  调研背景：「放量+创新高」经 verify 回测证明是右侧追高(截面 RankIC 显著为负)。吸筹的真正
  指纹是方向性的——参考 Wyckoff/VSA、A股筹码分布、OBV/ADL 三套体系，落地三个判别信号：

ambush 吸筹分（七项 0~100 原始分直接加权）：
  筹码  低位筹码集中                                  权重 0.20
  位置  价格中低位                                    权重 0.20
  CMF   高买压反向有效分（最高50）                     权重 0.10
  P3    缩量打压后首次完整收复                         权重 0.20
  P24   OBV底背离连续确认后的首次成立                   权重 0.10
  户数  股东户数下降                                  权重 0.10
  回购  近90日公司回购                                权重 0.10
旧四技术因子加权分及其一字板/派发/P20折扣已删除；出货风险只在机会分中单独折扣。

另叠加「游资形态匹配」(规格见 meta_data_backup/hot_money_patterns.md)：把游资坐庄的「吸筹→试盘→
洗盘→突破→拉升→出货」六段套路编码成 match_patterns() 的布尔匹配器，给每只票打形态标签，再由
_pattern_phase() 汇总成一个主导阶段（详见下表 + 该函数 docstring）。

─────────────────────────────────────────────────────────────────────────────
游资形态总表（PATTERNS，6 类 / 26 个，编号 P1-P26）
  · 列：编号  名称  —— 命中条件（位置=收盘价近60日分位；量比=近5日均量/前20日均量；
        漂移=近20日涨跌幅；CMF=Chaikin资金流；筹码集中=主峰±7%价带内筹码占比）。
  · 信号方向 buy/hold/sell；阶段配色按操作进程由早到晚渐变：吸筹🟢→试盘/洗盘🟡→突破/拉升🟠→出货🔴（观望⚪）。
  · 阶段优先级见 _pattern_phase：出货风险积分≥3 > 突破 > 吸筹/洗盘 > 拉升 > 试盘。
─────────────────────────────────────────────────────────────────────────────
【吸筹 🟢buy】主力在低位悄悄建仓
  P1 低位横盘磨人      位置<0.40 + 近20日振幅<18% + |漂移|<8%：低位窄幅横盘磨人
  P2 低位影线吸筹      位置<0.40 + 当日为十字星/长下影 + 近10日同类K线≥5根(下影>2×实体且>1.5%)
  P3 缩量打压首次收复  位置<0.35 + 近8日有"大阴跌≥4.5%且缩量(<0.85×此前20日均量)"，
                       期间不深破底并首次完整收复跌前收盘：隐性吸货确认
  P4 量增价稳吸收      位置<0.60 + 量比>1.2 + |漂移|<6% + CMF>0：放量但价稳、资金净流入
  P5 底部形态构筑      位置<0.45 + 近端两摆动低点等高/低点抬高(-4%~+8%) + 中间反弹≥6%(颈线) + 当前价回升未破颈线：双底/W底
  P23 箱体波动压缩     位置<0.60 + 近20日振幅<0.80×近60日：波动收窄蓄势
  P24 OBV底背离        位置<0.50 + 30日价≤+3% + OBV净流入>0.10 + 背离>0.25，连续5日后首次确认
  P25 低位横盘缩量后启动  120日低位 + 60日严格横盘缩量 + 量能启动 + 临近20日平台
  P26 获利盘风险            获利盘≥90%或5日前<40%且当前≥60%，并且当日量能≥此前20日均量1.5倍
【试盘 🟡hold】拉升前试探上方抛压
  P6 试盘长上影        近8日有长上影(>3%且>2×实体)创20日新高后收盘缩回；保留形态观测，不作为有效因子
  P7 底部异动放量      位置<0.40 + 量比>1.5 + 近5日最大振幅>7%；保留形态观测，不作为有效因子
【洗盘 🟡buy】震仓甩浮筹、不破结构
  P8 缩量回踩洗盘      站上MA20 + 回踩近10日高点-2%~-15% + 跌日量<涨日量 + 筹码集中≥0.40：挖坑不破位
  P9 边拉边洗          多头排列(MA5>10>20) + 近8日涨跌符号切换≥4次 + 低点抬高：边拉边洗
  P10 高换手洗盘       量比>1.5 + 收盘>MA20 + 筹码集中≥0.45：高换手震仓但筹码峰不发散
  P21 假跌破收回       近5日假摔穿前40日箱体下沿×0.985、收盘又站回：Wyckoff spring(各周期不显著,40日+1.15%弱)
【突破 🟠hold（阶段标签，优先级仅次出货）】放量右进
  P11 放量突破启动     右进左出触发器命中：从低/中位放量(>1.3×)突破近20日收盘高点；全历史复测为负向风险
【拉升 🟠hold】超短动量，持有期拉长需防反转
  P12 连板拉升         连续涨停 streak≥2；复测 2 日动量有效，10~40 日转为反转风险
  P13 首板卡位         今日首板(近20日无涨停) + 换手10%~45%；复测 2 日动量有效
【出货 🔴sell】高位派发，当风控示警
  P14 高位放量滞涨     位置≥0.85 + 量比>1.5 + 近5日涨幅≤2% + 上影>2%；复测 40 日风控有效
  P15 量价背离         创60日新高却缩量(量比<1.0)、或放量(>1.8)却5日涨幅<1%：量价背离顶
  P16 阴天量           位置≥0.80 + 近5日出现40日最大量且收阴；全历史复测 2~40 日风控有效
  P17 倒V反转          位置≥0.80 + 较前期冲高>15%后从峰值回落≤-8%；复测 2~40 日持续有效
  P18 顶部大阴包阳      位置≥0.80 + 昨阳今阴且今实体完全吞没昨实体：顶部看跌吞没
  P19 灌压巨量大阴      位置≥0.70 + 量比>1.8 + 实体跌>6%且收在当日价区下1/4；复测 5~40 日风控有效
  P20 均线放量破位      收盘跌破MA20(5日前还在MA20上方) + 量比>1.2；全历史复测 2~40 日风控有效
  P22 放量假突破        今日盘中破前40日高但收盘没站上 + 当日放量>1.8×；全历史复测 2~40 日风控有效

当前跨池核心有效集（2026-07-11 全历史逐日滚动，HAC + FDR）：
P3、P11、P12、P13、P14、P16、P17、P19、P20、P22、P24。
P6、P7 保留命中数据用于后续研究，但不作为有效因子或前端高亮。
P25 是实验买点，会在所有股票池计算，但不进入 PATTERN_EFFECTIVE。
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
         故低位股「吸筹分高却命中出货预警(P20)」是正确预警而非 bug——结构因子漏掉、形态层补上的真实风险。
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
     · 单池次级信号：旧P2/P24仅hotmoney中长期弱正；P15/P18仅hotmoney 20日负向。
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
       1,178/577，但HAC/FDR仍未通过。按后续使用要求，生产P23恢复原“位置<0.60 + 振幅比<0.80”状态规则；
       严格候选只保留在研究文件，不接入生产。
     · P24 改为位置<0.50、30日涨幅≤3%、量纲化OBV>0.10、OBV与价格背离>0.25，并连续5日后首次确认。
       命中由138,093/65,937降至1,144/677；10日超额+0.67%/+0.82%，绝对胜率56.5%/64.5%，双池10日
       均通过HAC与BH-FDR 10%，因此加入 PATTERN_EFFECTIVE。
  21) 【2026-07-11 P3首次收复优化】旧P3把信号日附近30日均量用于更早的下跌日，且一次下跌会连续多日重复命中；
     改为下跌日只和此前20日均量比较，并要求8日内首次完整收复跌前收盘、期间不深破底、确认日不过度放量。
     参数只用2017-2023开发段排序，2024+留出不参与选参；leader/hotmoney 两池始终使用同一规则。
     · 生产参数：位置<0.35、跌幅≥4.5%、下跌量比<0.85、8日内完整收复、最多再破底3%、确认日量比≤1.30。
     · 全历史命中843/605；2/5/10/20/40日超额 leader +0.44/+0.44/+0.64/+0.65/+0.37%，
       hotmoney +0.46/+1.04/+0.70/+0.92/+2.36%，十个池×周期全部为正。
     · leader 2日与 hotmoney 5日通过各池24×5次检验的BH-FDR 10%，因此P3重新加入 PATTERN_EFFECTIVE；
       仍存在当前成分股幸存者偏差，且未计交易成本与涨跌停可买性。
 22) 【2026-07-12 P25统一升级】按使用决定，将严格缩量底部方案作为全池统一 P25，不再区分
     leader/hotmoney 公式，并删除沪深300站上MA20门槛。判据为120日位置≤45%、60日振幅≤25%、60日绝对涨跌≤8%、近20日/前40日均量≤70%、
     60日量能斜率≤-1%、当日量≥前20日均量1.5倍、距20日平台≥-2%；保留防追高门槛。
     · leader 1,296次：2/5/10/20/40日同日池超额 -0.11%/-0.10%/-0.34%/-0.58%/-1.77%。
     · hotmoney 282次：2日约0%，5/10/20/40日 +0.39%/+0.63%/+1.51%/+1.46%，但FDR均未通过。
     · 结论：P25在所有池统一计算，继续作为实验标签，不加入 PATTERN_EFFECTIVE。
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import bisect
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


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CAPITAL_DIR = DATA_DIR / "capital"
DB_FILE = stock_storage.DEFAULT_DB_FILE
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
CHIP_WINNER_RISK_VOLUME_RATIO_MIN = 1.50  # P26：获利盘风险必须有放量确认
SEALED_AMP = 0.005       # 日内振幅 ≤ 0.5% 视为一字封死板
TURNOVER_COVERAGE = 0.7  # 近端窗口换手率覆盖率达标才用换手率，否则退回成交量
SUSPECT_ACCUM_SCORE = 65  # 无形态命中但吸筹分≥此值 → 疑似吸筹(待确认)；低于则观望
COMPRESS_AMP_RATIO = 0.80  # 近20日振幅 < 此倍×近60日振幅 → 波动压缩(酝酿)
PATTERN_CONFIRM_DAYS = 5   # P24：连续确认后只在首次成立日触发一次
OBV_POSITION_MAX = 0.50
OBV_PRICE_RETURN_MAX = 0.03
OBV_RETURN_MIN = 0.10      # P24：30日量纲化OBV净流入下限
OBV_DIV_MIN = 0.25         # P24：OBV净流入(量纲化)−价格涨幅的最小背离
P2_WINDOW = 10             # P2：近X日反复出现十字星/长下影（参数研究见纪要19）
P2_MIN_COUNT = 5           # P2：X日内至少Y次，且当日自身必须是其中一次；所有池参数相同
P25_POSITION_120_MAX = 0.45
P25_RANGE_60_MAX = 0.25
P25_ABS_RET_60_MAX = 0.08
P25_VOLUME_CONTRACT_MAX = 0.70
P25_VOLUME_SLOPE_60_MAX = -0.01
P25_VOLUME_IGNITION_MIN = 1.50
P25_BREAKOUT_20_MIN = -0.02

# 资金面维度（数据由 stock_crawl_holders.py / stock_crawl_capital.py 入库，见纪要(13)）：
#   股东户数下降/公司回购作为吸筹侧原始特征；龙虎榜上榜进出货侧避雷。
#   v3.3.x 起线上吸筹/出货总分都用原始特征直接加权，不再用截面 rank 做合成。
HOLDER_TABLE = "shareholder_count"     # code, disclose_date, change_pct(户数增减%)
REPURCHASE_TABLE = "repurchase"        # code, disclose_date
LHB_TABLE = "lhb_all"                  # code, date(上榜日)
CAPITAL_EVENT_DAYS = 90                # 回购/上榜：近 N 自然日内有事件视为"近期"
ACCUM_MODEL_WEIGHTS = {
    "chip": 0.2,                        # 低位筹码集中
    "position": 0.2,                    # 价格中低位
    "cmf_eff": 0.1,                     # CMF 反向有效分：高买压反转风险不加分
    "p3": 0.2,                          # P3 缩量阴线打压吸筹
    "p24": 0.1,                         # P24 OBV 底背离：连续确认后的首次成立日
    "holder_change": 0.1,               # 股东户数变化：户数降=高分，缺失=中性50
    "repurchase": 0.1,                  # 公司回购：近90日回购=100，否则0
}
ACCUM_FEATURES = tuple(ACCUM_MODEL_WEIGHTS.keys())
ACCUM_EXPERIMENT_HORIZONS = (5, 10, 20)
ACCUM_EXPERIMENT_GRID_STEP = 0.05
ACCUM_EXPERIMENT_MIN_WEIGHT = 0.1
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
    "p14": 0.10,          # 高位放量滞涨
    "p16": 0.10,          # 阴天量
    "p17": 0.15,          # 倒V反转
    "p19": 0.15,          # 灌压巨量大阴
    "p20": 0.05,          # 均线放量破位
    "p22": 0.05,          # 放量假突破
    "lhb_recent": 0.10,   # 近90日龙虎榜：反向避雷信号
    "technical": 0.15,    # 连续高位派发分：高位 + 20日涨幅 + 高位放量
    "divergence": 0.15,   # 原始 divergence 分，不用 div_eff
}
DIST_FEATURES = tuple(DIST_MODEL_WEIGHTS.keys())
DIST_EXPERIMENT_HORIZONS = (5, 10, 20)
DIST_EXPERIMENT_GRID_STEP = 0.1
DIST_EXPERIMENT_MIN_WEIGHT = 0.1
DIST_EXPERIMENT_SPREAD_W = 3.0
EXPERIMENT_VALIDATION_CANDIDATES = 200
OPPORTUNITY_DISTRIBUTION_PENALTY = 0.5
OPPORTUNITY_FORMULA = "accumulation_percentile * (1 - 0.5 * distribution_percentile / 100)"

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

# 出货预警采用形态积分制：弱风险需共振，P17/P19 强风险可单独触发。
DISTRIBUTION_WARNING_POINTS = {
    "P14": 2,
    "P16": 2,
    "P17": 3,
    "P19": 3,
    "P20": 1,
    "P22": 1,
    "P26": 3,
}
DISTRIBUTION_WARNING_THRESHOLD = 3

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
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
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


def _intraday_volume_projection(quote: Dict[str, Any]) -> Tuple[float, Optional[float], bool]:
    """盘中累计量不是全天量：按内置 U 型日内成交曲线折算，收盘后保持原值。"""
    quote_dt = _parse_quote_datetime(quote.get("quote_time"))
    if quote_dt is None:
        quote_date = str(quote.get("quote_date") or "")[:10]
        if quote_date == datetime.now().strftime("%Y-%m-%d"):
            now = datetime.now()
            quote_dt = datetime.strptime(f"{quote_date} {now:%H:%M:%S}", "%Y-%m-%d %H:%M:%S")
    if quote_dt is None:
        return 1.0, None, False
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


# ── 候选池：细分龙头(is_leader) / 游资小盘universe(is_hot_money) ──

POOLS = ("leader", "hotmoney")
DEFAULT_POOL = "leader"


def load_candidates(conn: sqlite3.Connection, pool: str = DEFAULT_POOL,
                    max_cap: Optional[float] = MAX_MARKET_CAP_YI) -> List[Dict[str, Any]]:
    """取候选池：pool='leader' 细分龙头(is_leader) / 'hotmoney' 游资小盘universe(is_hot_money，
    由 stock_crawl_hot_money_universe.py 建池)。

    max_cap>0 时剔除总市值超过该值(亿元)的股票(按市值分层查看小/中盘子集用)；
    市值缺失会先由 stock_storage 从 stock_history 最新非空 market_cap 回补。max_cap=None/0 不过滤。
    """
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
        })
    return out


def load_leader_candidates(conn: sqlite3.Connection,
                           max_cap: Optional[float] = MAX_MARKET_CAP_YI) -> List[Dict[str, Any]]:
    """细分龙头候选——load_candidates(pool='leader') 的兼容别名。"""
    return load_candidates(conn, DEFAULT_POOL, max_cap)


def _bar(row: sqlite3.Row) -> Dict[str, Any]:
    """把一行日线整成轻量短键 dict（解耦 DB 列名、PIT 切片更快）。"""
    return {
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
    chips = np.zeros(CHIP_BUCKETS)
    span = pmax - pmin
    for lo, hi, peak, t in zip(lows, highs, peaks, turns):
        frac = min(1.0, max(0.0, t / 100.0 * CHIP_DECAY))   # 当日搬移的筹码比例
        chips *= (1.0 - frac)                                # 旧筹码按换手衰减
        i0 = int((lo - pmin) / span * (CHIP_BUCKETS - 1))
        i1 = int((hi - pmin) / span * (CHIP_BUCKETS - 1))
        i0 = max(0, i0); i1 = min(CHIP_BUCKETS - 1, max(i0, i1))
        weights = _triangular_weights(grid[i0:i1 + 1], lo, hi, peak)
        chips[i0:i1 + 1] += frac * weights / weights.sum()
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


# ── 游资形态匹配器（P1-P26，规格见 meta_data_backup/hot_money_patterns.md）──────────
# 每个匹配器输入 (bars, ctx) 返回 bool（命中），PIT 安全（只用窗口内 bar）。
# 信号方向：buy=吸筹/洗盘(左侧) · hold=拉升(只标记不追) · sell=出货(风控/回避)。

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


def _build_pattern_context(code: str, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一次性算齐形态匹配要用的量价上下文（避免每个匹配器各算一遍）。"""
    current_bars = bars[-LOOKBACK:]
    closes = [b["close"] for b in current_bars]
    vol, _ = _volume_series(current_bars)
    _, pos = _score_position(current_bars)
    _, vol_ratio = _score_volume_ratio(vol)
    p26_volume_ratio = _p26_volume_ratio(vol)
    _, drift = _score_absorption(current_bars)
    _, cmf = _score_cmf(current_bars, vol)
    chip = _chip_metrics(current_bars)
    prior_end = len(bars) - CHIP_WINNER_RISK_DAYS
    prior_start = max(0, prior_end - LOOKBACK)
    prior_chip = _chip_metrics(bars[prior_start:prior_end]) if prior_end >= MIN_BARS else None
    chip_winner = chip.get("winner") if chip else None
    chip_winner_prior = prior_chip.get("winner") if prior_chip else None
    chip_winner_change = (
        chip_winner - chip_winner_prior
        if chip_winner is not None and chip_winner_prior is not None
        else None
    )
    sealed, streak = _sealed_and_streak(current_bars, code)
    triggered, _ = _breakout_trigger(current_bars, vol)
    ma = {n: _ma_last(closes, n) for n in (5, 10, 20, 60)}
    ma_bull = bool(ma[5] and ma[10] and ma[20] and ma[5] > ma[10] > ma[20]
                   and closes[-1] and closes[-1] > ma[5])
    return {
        "code": code, "closes": closes, "vol": vol,
        "pos": pos, "vol_ratio": vol_ratio, "drift": drift, "cmf": cmf,
        "chip": chip, "sealed": sealed, "streak": streak, "triggered": triggered,
        "chip_winner": chip_winner,
        "chip_winner_prior": chip_winner_prior,
        "chip_winner_change": chip_winner_change,
        "p26_volume_ratio": p26_volume_ratio,
        "ma5": ma[5], "ma10": ma[10], "ma20": ma[20], "ma60": ma[60], "ma_bull": ma_bull,
    }


# --- 吸筹 🟢buy ---
def _pat_low_consolidation(bars, ctx):           # P1 低位横盘磨人
    pos, drift, closes = ctx["pos"], ctx["drift"], ctx["closes"]
    if pos is None or drift is None:
        return False
    win = [c for c in closes[-20:] if c]
    if len(win) < 15:
        return False
    rng = (max(win) - min(win)) / (sum(win) / len(win))
    return pos < 0.40 and rng < 0.18 and abs(drift) < 0.08


def _pat_low_shadows(bars, ctx):                 # P2 低位十字星/长下影反复
    if ctx["pos"] is None or ctx["pos"] >= 0.40 or len(bars) < P2_WINDOW + 1:
        return False
    matched: List[bool] = []
    for i in range(len(bars) - P2_WINDOW, len(bars)):
        body, low = _kl_body(bars, i), _kl_lower(bars, i)
        matched.append(bool(_kl_doji(bars, i) or (body and low and low > 2 * body and low > 0.015)))
    return matched[-1] and sum(matched) >= P2_MIN_COUNT


def _pat_shakedown_absorb(bars, ctx):            # P3 隐性收集(缩量阴线打压吸筹)
    pos, vol, closes = ctx["pos"], ctx["vol"], ctx["closes"]
    if pos is None or pos >= 0.35 or len(bars) < 30:
        return False
    current_base = _mean([value for value in vol[-21:-1] if value])
    current_volume = vol[-1] if vol else None
    if not current_base or not current_volume or current_volume > 1.30 * current_base:
        return False
    for i in range(len(bars) - 2, len(bars) - 10, -1):
        chg, event_volume = bars[i]["chg"], vol[i]
        event_base_values = [value for value in vol[i - 20:i] if value]
        pre_drop, drop_close = closes[i - 1], closes[i]
        if (
            chg is None or chg > -4.5
            or not event_volume or not event_base_values
            or event_volume >= 0.85 * _mean(event_base_values)
            or not pre_drop or not drop_close or pre_drop <= drop_close
        ):
            continue
        # 必须在 8 日内首次完整收复跌前收盘；同一次打压不会连续多日重复命中。
        if not closes[-1] or not closes[-2] or closes[-1] < pre_drop or closes[-2] >= pre_drop:
            continue
        path = [value for value in closes[i:] if value]
        if len(path) != len(closes[i:]) or min(path) < drop_close * 0.97:
            continue
        return True
    return False


def _pat_absorption(bars, ctx):                  # P4 量增价稳(吸收)
    pos, vr, drift, cmf = ctx["pos"], ctx["vol_ratio"], ctx["drift"], ctx["cmf"]
    return (pos is not None and pos < 0.60 and vr is not None and vr > 1.2
            and drift is not None and abs(drift) < 0.06 and cmf is not None and cmf > 0)


def _pat_bottom_formation(bars, ctx):            # P5 底部形态构筑(双底/W底·低点抬高)
    """低位筑底：近端两个摆动低点等高或低点抬高、中间有像样反弹(颈线)，当前价正从二次探底
    回升但尚未显著突破颈线（突破后归 P11/拉升）。可选筹码不发散增强可信度。"""
    pos, closes, chip = ctx["pos"], ctx["closes"], ctx["chip"]
    if pos is None or pos >= 0.45:
        return False
    cs = [c for c in closes[-60:] if c]
    if len(cs) < 40:
        return False
    lows = _swing_lows(cs, k=3)
    if len(lows) < 2:
        return False
    i1, i2 = lows[-2], lows[-1]
    if i2 - i1 < 5:                               # 两底间隔太近不算结构
        return False
    l1, l2 = cs[i1], cs[i2]
    if not -0.04 <= l2 / l1 - 1 <= 0.08:          # 等高(±)或低点抬高，排除二次破位
        return False
    neck = max(cs[i1:i2 + 1])                     # 两底之间的反弹高点=颈线
    if neck / min(l1, l2) - 1 < 0.06:             # 中间反弹太弱，是平台不是双底
        return False
    c = cs[-1]
    if c <= l2 or c / neck - 1 > 0.03:            # 已回升但还没显著突破颈线(突破归拉升)
        return False
    return chip is None or chip["concentration"] >= 0.35


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


def _pat_compression(bars, ctx):                 # P23 箱体波动压缩(酝酿)
    """中低位 + 近20日振幅 < 0.80×近60日振幅：波动收窄、蓄势待发。"""
    pos = ctx["pos"]
    if pos is None or pos >= 0.60:
        return False
    ratio = _amp_ratio(bars)
    return ratio is not None and ratio < COMPRESS_AMP_RATIO


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


def _pat_bottom_base_ignition(bars, ctx):         # P25 低位横盘缩量后启动
    """120日低位 + 60日严格横盘缩量，随后量能启动并接近20日平台。

    统一使用严格缩量底部判据；所有股票池口径相同。
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

    volumes = [b.get("volume") for b in bars]
    if any(value is None or value <= 0 for value in volumes[-64:]):
        return False
    early40 = _mean(volumes[-60:-20])
    recent20 = _mean(volumes[-20:])
    previous20 = _mean(volumes[-21:-1])
    if not early40 or not recent20 or not previous20:
        return False
    if recent20 / early40 > P25_VOLUME_CONTRACT_MAX:
        return False
    if volumes[-1] / previous20 < P25_VOLUME_IGNITION_MIN:
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


def _pat_chip_winner_risk(bars, ctx):             # P26 获利盘风险
    """获利筹码拥挤/快速转盈且当日放量，提示兑现或派发风险。"""
    winner = ctx.get("chip_winner")
    prior = ctx.get("chip_winner_prior")
    volume_ratio = ctx.get("p26_volume_ratio")
    if volume_ratio is None or volume_ratio < CHIP_WINNER_RISK_VOLUME_RATIO_MIN:
        return False
    return bool(
        (winner is not None and winner >= CHIP_WINNER_RISK_HIGH)
        or (
            winner is not None and prior is not None
            and prior < CHIP_WINNER_RISK_PRIOR_MAX
            and winner >= CHIP_WINNER_RISK_CURRENT_MIN
        )
    )


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
    closes, vol, ma20, chip = ctx["closes"], ctx["vol"], ctx["ma20"], ctx["chip"]
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
    vr, ma20, closes, chip = ctx["vol_ratio"], ctx["ma20"], ctx["closes"], ctx["chip"]
    return (vr is not None and vr > 1.5 and ma20 and closes[-1] and closes[-1] > ma20
            and chip is not None and chip["concentration"] >= 0.45)


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


def _pat_vol_price_div(bars, ctx):               # P15 量价背离
    closes, vr = ctx["closes"], ctx["vol_ratio"]
    win = [c for c in closes[-60:] if c]
    if len(win) < 30 or vr is None or closes[-1] is None:
        return False
    if closes[-1] < max(win):
        return False
    r5 = _ret_k(closes, 5)
    return vr < 1.0 or (vr > 1.8 and r5 is not None and r5 < 0.01)


def _pat_bearish_max_vol(bars, ctx):             # P16 阴天量(近期最大量收阴)
    pos, vol = ctx["pos"], ctx["vol"]
    if pos is None or pos < 0.80 or len(bars) < 40:
        return False
    mx = max([v for v in vol[-40:] if v] or [0])
    for i in range(len(bars) - 5, len(bars)):
        if vol[i] and vol[i] >= mx and bars[i]["chg"] is not None and bars[i]["chg"] < 0:
            return True
    return False


def _pat_inverted_v(bars, ctx):                  # P17 倒V反转
    pos, closes = ctx["pos"], ctx["closes"]
    if pos is None or pos < 0.80 or len(bars) < 16:
        return False
    recent = [c for c in closes[-10:] if c]
    base = [c for c in closes[-16:-10] if c]
    if len(recent) < 8 or not base:
        return False
    peak = max(recent)
    return (peak / min(base) - 1 > 0.15) and (closes[-1] / peak - 1 <= -0.08)


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
    """均线放量破位判定：收盘跌破MA20 + 5日前还在MA20上方 + 量比>1.2。
    P20 形态与吸筹分的「破位惩罚」共用此单一判据，避免两处逻辑漂移。"""
    ma20 = _ma_last(closes, 20)
    if ma20 is None or len(closes) < 25 or vol_ratio is None:
        return False
    c = closes[-1]
    if not c or c >= ma20:
        return False
    prior, ma20_prior = closes[-5], _ma_last(closes[:-4], 20)
    return bool(prior and ma20_prior and prior > ma20_prior and vol_ratio > 1.2)


def _pat_ma_breakdown(bars, ctx):                # P20 均线放量破位
    return _ma_breakdown(ctx["closes"], ctx["vol_ratio"])


def _pat_spring_reclaim(bars, ctx):              # P21 假跌破收回(Wyckoff spring·挖坑收回)
    """近5日最低跌穿前40日箱体下沿×0.985(假摔)，但当前收盘又站回箱体下沿之上。沿用早期形态研究。"""
    if len(bars) < 45:
        return False
    lows = [b["low"] for b in bars if b["low"] is not None]
    if len(lows) < 45:
        return False
    box_low_prior = min(lows[-45:-5])            # 前40日(不含近5日)箱体下沿
    recent_low = min(lows[-5:])
    c = ctx["closes"][-1]
    if not box_low_prior or not recent_low or not c:
        return False
    return recent_low < box_low_prior * 0.985 and c > box_low_prior


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
    ("P1", "低位横盘磨人", "吸筹", "buy", _pat_low_consolidation),
    ("P2", "低位影线吸筹", "吸筹", "buy", _pat_low_shadows),
    ("P3", "缩量打压首次收复", "吸筹", "buy", _pat_shakedown_absorb),
    ("P4", "量增价稳吸收", "吸筹", "buy", _pat_absorption),
    ("P5", "底部形态构筑", "吸筹", "buy", _pat_bottom_formation),
    ("P6", "试盘长上影", "试盘", "hold", _pat_test_upper_shadow),
    ("P7", "底部异动放量", "试盘", "hold", _pat_bottom_spike),
    ("P8", "缩量回踩洗盘", "洗盘", "buy", _pat_pullback_shakeout),
    ("P9", "边拉边洗", "洗盘", "buy", _pat_climb_wash),
    ("P10", "高换手洗盘", "洗盘", "buy", _pat_high_turnover_wash),
    ("P11", "放量突破启动", "突破", "hold", _pat_breakout),
    ("P12", "连板拉升", "拉升", "hold", _pat_consecutive_limit),
    ("P13", "首板卡位", "拉升", "hold", _pat_first_board),
    ("P14", "高位放量滞涨", "出货", "sell", _pat_high_vol_stall),
    ("P15", "量价背离", "出货", "sell", _pat_vol_price_div),
    ("P16", "阴天量", "出货", "sell", _pat_bearish_max_vol),
    ("P17", "倒V反转", "出货", "sell", _pat_inverted_v),
    ("P18", "顶部大阴包阳", "出货", "sell", _pat_bearish_engulf),
    ("P19", "灌压巨量大阴", "出货", "sell", _pat_dump_bigbear),
    ("P20", "均线放量破位", "出货", "sell", _pat_ma_breakdown),
    ("P21", "假跌破收回", "洗盘", "buy", _pat_spring_reclaim),       # 当前统一复测未纳入有效集
    ("P22", "放量假突破", "出货", "sell", _pat_failed_breakout),     # 全历史逐日复测：双池2~40日风控有效
    ("P23", "箱体波动压缩", "吸筹", "buy", _pat_compression),        # 状态型观察标签，不进入核心有效集
    ("P24", "OBV底背离", "吸筹", "buy", _pat_obv_divergence),        # 强OBV背离，连续5日后一次性确认
    ("P25", "低位横盘缩量后启动", "吸筹", "buy", _pat_bottom_base_ignition),
    ("P26", "获利盘风险", "出货", "sell", _pat_chip_winner_risk),
]

# 每个形态的命中条件一句话（供前端「命中形态解释」模块；与文件头总表口径一致）。
PATTERN_DESC: Dict[str, str] = {
    "P1": "位置<0.40 + 近20日振幅<18% + |20日涨跌|<8%：低位窄幅横盘磨人",
    "P2": "位置<0.40 + 当日为十字星/长下影 + 近10日合计≥5根：反复探底当日再次确认",
    "P3": "位置<0.35 + 近8日缩量大阴跌≥4.5%，不深破底且首次完整收复跌前收盘；最新复测leader 2日/hotmoney 5日通过FDR",
    "P4": "位置<0.60 + 量比>1.2 + |20日涨跌|<6% + CMF>0：放量但价稳、资金净流入",
    "P5": "位置<0.45 + 近端两摆动低点等高/低点抬高 + 中间反弹≥6%(颈线)：双底/W底",
    "P6": "近8日长上影(>3%且>2×实体)创20日新高后收盘缩回：探顶又压回；保留形态观测，不作为有效因子",
    "P7": "位置<0.40 + 量比>1.5 + 近5日最大振幅>7%：低位突然放量异动；保留形态观测，不作为有效因子",
    "P8": "站上MA20 + 回踩近10日高点-2%~-15% + 跌日量<涨日量 + 筹码集中≥0.40：挖坑不破位",
    "P9": "多头排列(MA5>10>20) + 近8日涨跌切换≥4次 + 低点抬高：边拉边洗",
    "P10": "量比>1.5 + 收盘>MA20 + 筹码集中≥0.45：高换手震仓但筹码峰不发散",
    "P11": "右进左出触发器：从低/中位放量(>1.3×)突破近20日收盘高点；全历史逐日复测为双池负向风险",
    "P12": "连续涨停 streak≥2；全历史逐日复测：2日动量有效，hotmoney 池10~40日转为反转风险",
    "P13": "今日首板(近20日无涨停) + 换手10%~45%；全历史逐日复测：2日动量有效，20~40日转为反转风险",
    "P14": "位置≥0.85 + 量比>1.5 + 近5日涨幅≤2% + 上影>2%：高位放量不涨；全历史逐日复测：双池2~40日风控有效",
    "P15": "创60日新高却缩量(量比<1.0)、或放量(>1.8)却5日涨幅<1%：量价背离顶",
    "P16": "位置≥0.80 + 近5日出现40日最大量且收阴：天量收阴；全历史逐日复测：双池2~40日风控有效",
    "P17": "位置≥0.80 + 较前期冲高>15%后从峰值回落≤-8%：冲高倒V；全历史逐日复测：双池2~40日持续负超额",
    "P18": "位置≥0.80 + 昨阳今阴且今实体完全吞没昨实体：顶部看跌吞没",
    "P19": "位置≥0.70 + 量比>1.8 + 实体跌>6%且收在当日价区下1/4：巨量灌压大阴；全历史逐日复测：双池2~40日持续负超额",
    "P20": "收盘跌破MA20(5日前还在MA20上方) + 量比>1.2：放量破位；全历史逐日复测：双池2~40日风控有效",
    "P21": "近5日最低假摔穿前40日箱体下沿×0.985、收盘又站回：假跌破收回(Wyckoff spring)",
    "P22": "今日盘中破前40日高但收盘没站上 + 当日放量>1.8×：放量假突破(高位拒绝)；全历史逐日复测：双池2~40日风控有效",
    "P23": "中低位 + 近20日振幅<0.80×近60日：箱体波动压缩、蓄势酝酿",
    "P24": "位置<0.50 + 30日涨幅≤3% + 量纲化OBV>0.10 + OBV价差>0.25，连续5日后仅首次确认；最新复测双池10日有效",
    "P25": "120日价格分位≤45% + 60日振幅≤25%且涨跌≤8% + 近20日/前40日均量≤70% + 量能斜率≤-1% + 当日量≥前20日均量1.5倍 + 距20日平台≤2%；所有池统一计算",
    "P26": "获利盘比例≥90%，或5个交易日前<40%且当前≥60%，并要求当日量能≥此前20日均量1.5倍：获利筹码拥挤/快速转盈后的放量兑现风险；全历史复测双池2~40日均通过HAC与FDR，所有池统一计算",
}

# 2026-07-12 全历史逐日滚动、双池均通过 HAC + BH FDR 的核心有效形态（前端高亮）：
#   P12/P13 为超短动量，P24 为10日吸筹买点，P26为获利盘兑现风险，其余为负向风险/出货风控；
#   P25 作全池实验买点。
PATTERN_EFFECTIVE = {"P3", "P11", "P12", "P13", "P14", "P16", "P17", "P19", "P20", "P22", "P24", "P26"}
PATTERN_EFFECTIVE_STYLE = {
    "P3": "bullish",
    "P11": "risk",
    "P12": "momentum",
    "P13": "momentum",
    "P14": "risk",
    "P16": "risk",
    "P17": "risk",
    "P19": "risk",
    "P20": "risk",
    "P22": "risk",
    "P24": "bullish",
    "P26": "risk",
}


def pattern_catalog() -> List[Dict[str, Any]]:
    """形态总表结构化输出，含后端统一维护的有效性与前端颜色语义。"""
    return [
        {
            "code": c, "name": n, "category": cat, "signal": sig,
            "desc": PATTERN_DESC.get(c, ""), "effective": c in PATTERN_EFFECTIVE,
            "effective_style": PATTERN_EFFECTIVE_STYLE.get(c, "neutral"),
        }
        for c, n, cat, sig, _ in PATTERNS
    ]


def match_patterns(code: str, bars: List[Dict[str, Any]],
                   ctx: Optional[Dict[str, Any]] = None,
                   pool: Optional[str] = None) -> List[Dict[str, str]]:
    """对一段日线窗口匹配全部形态，返回命中列表（PIT 安全）。"""
    if len(bars) < MIN_BARS:
        return []
    ctx = ctx or _build_pattern_context(code, bars)
    current_bars = bars[-LOOKBACK:]
    fired: List[Dict[str, str]] = []
    for pcode, name, phase, signal, fn in PATTERNS:
        try:
            if fn(current_bars, ctx):
                fired.append({"code": pcode, "name": name, "phase": phase, "signal": signal})
        except Exception:
            continue
    return fired


def _distribution_warning_points(fired: Sequence[Dict[str, str]]) -> int:
    """命中出货形态的累计预警积分；同一形态最多计一次。"""
    codes = {str(pattern.get("code") or "") for pattern in fired}
    return sum(DISTRIBUTION_WARNING_POINTS.get(code, 0) for code in codes)


def _pattern_phase(fired: List[Dict[str, str]], score: Optional[float] = None) -> str:
    """命中形态汇总成主导阶段（优先级：出货积分达标 > 突破 > 买入区 > 拉升 > 试盘）。

    出货预警只统计 P14/P16/P17/P19/P20/P22，累计达到
    DISTRIBUTION_WARNING_THRESHOLD 才触发；P15/P18 保留观察但不计预警积分。
    突破=放量右进买点，最 actionable，仅次于达标的出货风控示警、优先于被动的吸筹/洗盘；
    买入区按类别细分吸筹 / 洗盘（两类都中则合并标注）；剩余 hold 区拉升中 > 试盘；
    无任何形态命中：吸筹分≥SUSPECT_ACCUM_SCORE → 疑似吸筹(待确认)，否则 → 观望(场外不参与)。
    """
    sigs = {p["signal"] for p in fired}
    cats = {p["phase"] for p in fired}
    if _distribution_warning_points(fired) >= DISTRIBUTION_WARNING_THRESHOLD:
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
    if score is not None and score >= SUSPECT_ACCUM_SCORE:
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
        "p16": 1.0 if "P16" in codes else 0.0,
        "p17": 1.0 if "P17" in codes else 0.0,
        "p19": 1.0 if "P19" in codes else 0.0,
        "p20": 1.0 if "P20" in codes else 0.0,
        "p22": 1.0 if "P22" in codes else 0.0,
        "lhb_recent": 1.0 if lhb_recent else 0.0,
        "technical": _clip01((technical_distribution_score or 0.0) / 100.0),
        "divergence": _clip01((divergence_score or 0.0) / 100.0),
    }


def _distribution_model_score(features: Dict[str, float],
                              weights: Optional[Dict[str, float]] = None) -> float:
    weights = weights or DIST_MODEL_WEIGHTS
    return round(100.0 * sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in weights), 1)


def _apply_distribution_model(row: Dict[str, Any]) -> None:
    signals = row.setdefault("signals", {})
    pattern_codes = row.get("patterns") or []
    features = _distribution_model_features(
        signals.get("technical_distribution_score", row.get("distribution_score")),
        pattern_codes,
        bool(row.get("lhb_recent")),
        (row.get("sub_scores") or {}).get("divergence"),
    )
    score = _distribution_model_score(features)
    row["distribution_score"] = score
    signals["distribution_score"] = score
    signals["distribution_model_features"] = {k: round(v, 4) for k, v in features.items()}
    signals["distribution_model_weights"] = {k: round(float(DIST_MODEL_WEIGHTS.get(k, 0.0)), 4) for k in DIST_FEATURES}


def _score_bars(code: str, bars: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """提取雷达需要的技术原始分与形态上下文；数据不足返回 None。

    这里继续计算 position/chip/cmf_eff 等原始特征；chip 同时供吸筹总分、展示与形态匹配。
    不再生成旧四技术因子加权分。
    最终吸筹分统一由 ``_accumulation_model_score`` 计算。
    """
    if len(bars) < MIN_BARS:
        return None
    vol, vol_measure = _volume_series(bars)
    s_pos, close_pctile = _score_position(bars)            # 位置
    s_div, drift = _score_divergence(bars, vol)            # ① 努力与结果背离
    s_cmf, cmf = _score_cmf(bars, vol)                     # ② 收盘买压
    s_chip, chip = _score_chip(bars)                       # ③ 低位筹码集中
    _, vol_ratio = _score_volume_ratio(vol)               # 仅供展示
    sealed, streak = _sealed_and_streak(bars, code)
    triggered, trigger = _breakout_trigger(bars, vol)     # 右进左出触发器

    # CMF 在七因子模型中反向计入：高买压是反转风险，低/中性买压最多给50分。
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


def _state_label(score: Optional[float], sealed: int, streak: int, triggered: bool = False) -> str:
    if sealed > 0 or streak >= 4:
        return "已启动(封板/连板,非吸筹)"
    if score is None:
        return "数据不足"
    if triggered and score >= 40:
        return "放量突破(右进)"      # 事件标记；verify 显示截面不占优，慎追
    if score >= 65:
        return "疑似吸筹"
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
      - 出货：看 sell 形态 + 出货分，吸筹分高则略扣；
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
        shape = _phase_pattern_strength(fired, {"sell"})
        conflict_penalty = max(0.0, ambush - dist) * 0.12
        return round(_clip_score(0.55 * shape + 0.45 * dist - conflict_penalty), 1)

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
    pos = sig.get("close_pctile")
    if bars is not None and pos is not None and pos < 0.60:
        ratio = _amp_ratio(bars)
        if ratio is not None and ratio < COMPRESS_AMP_RATIO:
            add("波动压缩(酝酿)", "bullish")
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
    if "P26" in {pattern.get("code") for pattern in fired}:
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


def _score_candidate_from_bars(cand: Dict[str, Any], bars: List[Dict[str, Any]],
                               pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    """给单只候选按给定 bars 打分；离线/实时路径共用同一套形态逻辑。"""
    out = dict(cand)
    current_bars = bars[-LOOKBACK:]
    res = _score_bars(cand["code"], current_bars)
    if res is None:
        out.update({"ambush_score": None, "score_status": "INSUFFICIENT_DATA",
                    "state": "数据不足", "last_date": bars[-1]["date"] if bars else None})
        return out
    pattern_context = _build_pattern_context(cand["code"], bars)
    fired = match_patterns(cand["code"], bars, ctx=pattern_context, pool=pool)
    winner = pattern_context.get("chip_winner")
    winner_prior = pattern_context.get("chip_winner_prior")
    winner_change = pattern_context.get("chip_winner_change")
    res["signals"].update({
        "chip_winner": round(winner, 2) if winner is not None else None,
        "chip_winner_5d_ago": round(winner_prior, 2) if winner_prior is not None else None,
        "chip_winner_5d_change": round(winner_change, 3) if winner_change is not None else None,
        "p26_volume_ratio": round(pattern_context["p26_volume_ratio"], 2)
        if pattern_context.get("p26_volume_ratio") is not None else None,
        "chip_winner_risk": bool("P26" in {pattern["code"] for pattern in fired}),
    })
    pattern_codes = [p["code"] for p in fired]
    distribution_warning_points = _distribution_warning_points(fired)
    res["patterns"] = pattern_codes
    _apply_distribution_model(res)
    score_features = {
        "sub_scores": res.get("sub_scores") or {},
        "patterns": pattern_codes,
        "holder_change": out.get("holder_change"),
        "repurchase_recent": bool(out.get("repurchase_recent")),
    }
    score = _accumulation_model_score(_accumulation_raw_features(score_features))
    phase = _pattern_phase(fired, score)
    out.update({
        "ambush_score": score,
        "distribution_score": res["distribution_score"],
        "score_status": "OK",
        "triggered": res["triggered"],
        "state": _state_label(score, res["sealed"], res["streak"], res["triggered"]),
        "last_date": bars[-1]["date"],
        "patterns": pattern_codes,
        "pattern_detail": fired,
        "distribution_warning_points": distribution_warning_points,
        "pattern_phase": phase,
        "phase_confidence": _phase_confidence(phase, fired, score, res["distribution_score"]),
        "invalidations": _phase_invalidations(phase),
        "evidence": _evidence(res, fired, current_bars),
        "signals": res["signals"],
        "sub_scores": res["sub_scores"],
    })
    out["signals"]["distribution_warning_points"] = distribution_warning_points
    out["signals"]["distribution_warning_threshold"] = DISTRIBUTION_WARNING_THRESHOLD
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


def score_candidate(conn: sqlite3.Connection, cand: Dict[str, Any],
                    as_of: Optional[str] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    """给单只龙头算当下潜伏分 + 游资形态匹配，返回带子分/原始信号/形态的明细行。

    as_of 给定时只用该日期及以前的 bar（PIT 防泄漏，供历史复盘）。
    pool='hotmoney' 且开启平滑(REVERSAL_SMOOTH_DAYS>1)时附 rev_hist(近 N 日原始过热因子,供 EMA 平滑)。
    """
    bars = _recent_bars(conn, cand["code"], limit=LOOKBACK + CHIP_WINNER_RISK_DAYS, as_of=as_of)
    return _score_candidate_from_bars(cand, bars, pool=pool)


# ── 输出 ──────────────────────────────────────────────────────

def base_payload(mode: str, candidate_count: int) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "sw3_member.is_leader (细分行业龙头池)",
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
        "p3": 100.0 if "P3" in pattern_codes else 0.0,
        "p24": 100.0 if "P24" in pattern_codes else 0.0,
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


def _apply_accumulation_model(rows: List[Dict[str, Any]]) -> None:
    """吸筹/出货保留原始解释分；机会分使用横截面百分位合成。"""
    for r in rows:
        signals = r.setdefault("signals", {})
        features = _accumulation_raw_features(r)
        score = _accumulation_model_score(features)
        r["ambush_score"] = score
        signals["accumulation_model_features"] = {k: round(v, 1) for k, v in features.items()}
        signals["accumulation_model_weights"] = {k: round(float(ACCUM_MODEL_WEIGHTS.get(k, 0.0)), 4) for k in ACCUM_FEATURES}
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
        ev.append({"label": "近期上龙虎榜(避雷)", "kind": "bearish"})


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

# 潜伏妖股筛选(latent 模式，研究见 meta_data_backup/hot_money_latent_hunting_research.md)：
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


MARKET_REGIME_INDEX = "510310"   # 沪深300 ETF(index_nav)；优先用累计净值 MA20 判趋势(纪要14择时)


def _market_regime(conn: sqlite3.Connection, as_of: Optional[str] = None) -> Dict[str, Any]:
    """市场状态(PIT)：沪深300 收盘 vs MA20。纪要(14)：反转分仅在大盘>MA20 时做多收益翻倍(+0.83% vs +0.47%)。

    ETF 单位净值(nav)会受分红/折算影响出现跳变；择时只需要连续价格代理，因此优先用累计净值(nav_acc)。
    """
    entry = stock_storage.load_index_nav(conn, MARKET_REGIME_INDEX)
    recs = (entry.get("records") if isinstance(entry, dict) else None) or []
    series = []
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
    if len(series) < 21:
        return {"available": False}
    series.sort()
    closes = [v for _, v, _ in series]
    last_date, last, last_field = series[-1]
    ma20 = sum(closes[-20:]) / 20.0
    above = last > ma20
    ret5 = (closes[-1] / closes[-6] - 1.0) if len(closes) > 5 else None
    return {
        "available": True, "index": MARKET_REGIME_INDEX, "date": last_date,
        "value_field": last_field, "value": round(last, 4), "ma20": round(ma20, 4),
        "fallback_count": fallback_count,
        "above_ma20": above, "ret5": round(ret5, 4) if ret5 is not None else None,
        "favorable": bool(above),
        "note": ("大盘站上MA20：适合做多反转分(top档3日≈+0.83%)"
                 if above else "大盘跌破MA20：反转分做多收益骤降(≈+0.13%,接刀)，宜观望/对冲"),
    }


def _quote_to_realtime_bar(bars: List[Dict[str, Any]], quote: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    price = _safe(quote.get("price"))
    quote_date = str(quote.get("quote_date") or "")[:10]
    if price is None or not quote_date or not bars:
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
    volume_factor, volume_elapsed, volume_projected = _intraday_volume_projection(quote)

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

    codes = [str(row.get("code")).zfill(6) for row in stocks]
    conn = stock_storage.connect(DB_FILE)
    try:
        bars_by_code = _bulk_recent_bars(conn, codes, LOOKBACK + CHIP_WINNER_RISK_DAYS)
    finally:
        conn.close()

    pool = str(base.get("pool") or DEFAULT_POOL)
    quote_sources = Counter(str(quote.get("source") or "unknown") for quote in quotes.values())
    primary_source = quote_sources.most_common(1)[0][0] if quote_sources else "realtime_batch"
    rescored: List[Dict[str, Any]] = []
    matched_quotes = 0
    used_realtime = 0
    missing_history = 0
    for old in stocks:
        code = str(old.get("code")).zfill(6)
        bars = bars_by_code.get(code) or []
        quote = quotes.get(code)
        if quote:
            matched_quotes += 1
        merged_bars, used_quote = _merge_realtime_quote_bars(
            bars, quote, LOOKBACK + CHIP_WINNER_RISK_DAYS
        )
        if not merged_bars:
            missing_history += 1
            row = dict(old)
            row["realtime_status"] = "NO_LOCAL_BARS"
            rescored.append(row)
            continue
        row = _score_candidate_from_bars(old, merged_bars, pool=pool)
        if quote:
            row["realtime_price"] = quote.get("price")
            row["realtime_change_pct"] = quote.get("change_pct")
            row["realtime_quote_time"] = quote.get("quote_time")
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
            if quote.get("market_cap_yi") is not None:
                row["market_cap_yi"] = quote.get("market_cap_yi")
        row["realtime_status"] = "UPDATED" if used_quote else "LOCAL_BARS_ONLY"
        if used_quote:
            used_realtime += 1
        _apply_distribution_model(row)
        _attach_capital_evidence(row)
        rescored.append(row)

    ranked = [row for row in rescored if row.get("ambush_score") is not None]
    _apply_accumulation_model(ranked)
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
            "note": "批量实时行情只用于当前页面实时重算，不写回离线结果文件。",
        },
    })
    return base


def run_ambush(as_of: Optional[str] = None,
               max_cap: Optional[float] = None,
               pool: str = DEFAULT_POOL,
               write: bool = True,
               print_summary: bool = True) -> Dict[str, Any]:
    """write/print_summary=False 时只返回打分 payload、不落盘不打印（供 latent 模式复用打分核心）。"""
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        scored = [score_candidate(conn, cand, as_of=as_of, pool=pool) for cand in candidates]
        capital_map, capital_available = _load_capital_map(conn, as_of)
        market_regime = _market_regime(conn, as_of) if pool == "hotmoney" else None
    finally:
        conn.close()

    # 资金面维度：股东户数↓/回购=吸筹侧原始分、龙虎榜上榜=出货侧避雷。
    for r in scored:
        info = capital_map.get(r["code"])
        r["holder_change"] = info.get("holder_change") if info else None
        r["repurchase_recent"] = bool(info and info.get("repurchase_recent"))
        r["lhb_recent"] = bool(info and info.get("lhb_recent"))
        _apply_distribution_model(r)
        _attach_capital_evidence(r)

    # PIT 防泄漏：theme_candidates.json 是「最新日」快照，历史 as-of 复盘时用它会泄露未来题材热度，
    # 故只在 as_of=None（实时跑）时才挂题材/行业热度映射；历史复盘留空，避免未来热度泄漏。
    if as_of is None:
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
    _apply_accumulation_model(ranked)
    if pool == "hotmoney":
        _apply_reversal_model(ranked)              # 游资池主排序=超短反转分(纪要14)
        ranked.sort(key=lambda r: (r.get("reversal_score") or 0.0), reverse=True)
    else:
        ranked.sort(key=lambda r: (r.get("opportunity_score") or 0.0, r.get("ambush_score") or 0.0), reverse=True)
    skipped = len(scored) - len(ranked)

    payload = base_payload("ambush", len(candidates))
    payload.update({
        "status": "ok" if candidates else "empty",
        "description": "细分龙头机会分：吸筹分、出货分展示原始模型分；仅计算机会分时把二者转为候选池百分位后做出货风险折扣。",
        "params": {
            "lookback": LOOKBACK,
            "cmf_full": CMF_FULL, "chip_band": CHIP_BAND, "pos_low": POS_LOW, "pos_high": POS_HIGH,
            "accumulation_model": {
                "weights": ACCUM_MODEL_WEIGHTS,
                "features": list(ACCUM_FEATURES),
                "missing_holder_change_score": 50.0,
                "experiment_file": display_path(ACCUM_EXPERIMENT_FILE),
            },
            "distribution_model": {
                "weights": DIST_MODEL_WEIGHTS,
                "features": list(DIST_FEATURES),
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
        "max_market_cap_yi": max_cap,
        "theme_source": theme_meta,
        "capital_available": capital_available,
        "capital_counts": {
            "holder_down": sum(1 for r in ranked if (r.get("holder_change") or 0) < 0),
            "repurchase": sum(1 for r in ranked if r.get("repurchase_recent")),
            "lhb_avoid": sum(1 for r in ranked if r.get("lhb_recent")),
        },
        "stocks": ranked,
    })
    if not candidates:
        payload["notes"] = ["候选池为空：先运行 python stock_crawl_segment_leaders.py crawl 选龙头并回写 is_leader。"]
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
    pool_note = f"≤{cap:g}亿小中盘龙头" if cap else "全市值龙头(含大盘)"
    print(f"  生成时间: {payload['generated_at']}{as_of_note} · 候选({pool_note}): "
          f"{payload['candidate_count']} · 已打分: {payload.get('scored_count', 0)}")
    print(f"  阶段分布(游资操作顺序): {dist}")
    cc = payload.get("capital_counts") or {}
    if payload.get("capital_available"):
        print(f"  资金面(纪要13): 户数下降{cc.get('holder_down',0)} · 回购中{cc.get('repurchase',0)}"
              f" · 上龙虎榜避雷{cc.get('lhb_avoid',0)} · 户数/回购已并入吸筹总分 · 排序=机会分")
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
    """对每只候选滑动取历史截面，PIT 重算七因子吸筹分并配对未来前向收益。

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
            fired = match_patterns(code, window, pool=pool)
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
    payload = base_payload("verify", len(candidates))
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "七因子吸筹分后验回测：PIT 重算吸筹分 vs 未来前向收益（分位单调性 / 截面RankIC / 多空价差）。",
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
    series: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, int]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + CHIP_WINNER_RISK_DAYS + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})
    if not series:
        return {"samples": [], "dates": [], "codes": []}

    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    as_of_dates = _verify_as_of_dates(all_dates[:-max_h])
    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    for d in as_of_dates:
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK + CHIP_WINNER_RISK_DAYS - 1 or i + max_h >= len(bars):
                continue
            window = bars[i - LOOKBACK - CHIP_WINNER_RISK_DAYS + 1:i + 1]
            fired = match_patterns(code, window, pool=pool)
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


def _horizon_effective(signal: str, hz: Dict[str, Any]) -> bool:
    """单个(形态,周期)是否有效：buy 超额>0且t≥1.5；sell 超额<0且t≤-1.5；hold 仅看显著(|t|≥1.5)。"""
    em, t = hz.get("excess_mean"), hz.get("excess_t_stat")
    if em is None or t is None:
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
    for grp in by_date.values():
        hit = [g["rets"][h] for g in grp if pcode in g["fired"]]
        if not hit:
            continue
        section_mean = _mean([g["rets"][h] for g in grp])
        excess.append(_mean(hit) - section_mean)
        hit_rets.extend(hit)
    if not hit_rets:
        return {"n_hits": 0, "n_sections": 0, "excess_mean": None, "excess_t_stat": None, "win_rate": None}
    em = _mean(excess)
    es = (sum((x - em) ** 2 for x in excess) / len(excess)) ** 0.5 if len(excess) > 1 else None
    t = (em / es * math.sqrt(len(excess))) if es else None
    return {
        "n_hits": len(hit_rets),
        "n_sections": len(excess),
        "excess_mean": round(em, 4),
        "excess_t_stat": round(t, 2) if t is not None else None,
        "win_rate": round(sum(1 for x in hit_rets if x > 0) / len(hit_rets), 3),
    }


def run_patterns(max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_pattern_samples(conn, candidates, pool=pool)
    finally:
        conn.close()

    samples = collected["samples"]
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_date.setdefault(s["date"], []).append(s)

    results = []
    for pcode, name, phase, signal, _fn in PATTERNS:
        horizons = {str(h): _pattern_event_study(by_date, pcode, h) for h in VERIFY_HORIZONS}
        row = {"code": pcode, "name": name, "phase": phase, "signal": signal,
               "horizons": horizons,
               "effective_at": [h for h in VERIFY_HORIZONS if _horizon_effective(signal, horizons[str(h)])]}
        results.append(row)

    payload = base_payload("patterns", len(candidates))
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "游资形态预测力后验：每个形态命中后，未来 N 日相对同日全体的超额收益（剔除大盘 beta）。",
        "params": {"horizons": list(VERIFY_HORIZONS), "step": VERIFY_STEP, "window_days": VERIFY_WINDOW_DAYS},
        "section_count": len(collected["dates"]),
        "sample_count": len(samples),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "patterns": results,
    })
    if not samples:
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
    print(f"  逐形态 × 逐周期超额收益(剔beta)；带 * = 该周期有效(buy超额>0且t≥1.5 / sell超额<0且t≤-1.5 / hold仅|t|≥1.5)")
    print(f"  {_ljust('形态', 6)}{_ljust('名称', 17)}{_ljust('阶段', 5)}{_ljust('信号', 5)}{_rjust('命中', 6)}  "
          + "".join(_rjust(f"{h}日", 9) for h in VERIFY_HORIZONS) + _rjust('有效@', 12))
    rows = sorted(results, key=lambda r: (r["horizons"][mid].get("excess_mean") or 0), reverse=True)
    for r in rows:
        hz = r["horizons"]
        cells = ""
        eff: List[str] = []
        for h in VERIFY_HORIZONS:
            d = hz[str(h)]
            ok = _horizon_effective(r["signal"], d)
            cells += f"{(_pct(d.get('excess_mean')) + ('*' if ok else '')):>9}"
            if ok:
                eff.append(str(h))
        eff_str = (",".join(eff) + "日") if eff else "—"
        print(f"  {r['code']:<6}{_ljust(r['name'], 17)}{_ljust(r['phase'], 5)}{r['signal']:<5}"
              f"{hz[mid].get('n_hits', 0):>6}  {cells}{_rjust(eff_str, 12)}")
    print("-" * 108)
    print("  有效@ = 该形态达标的持有期(交易日)。buy 看正超额、sell 看负超额、hold 看显著；均要 |t|≥1.5。")
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
    units = int(round(1.0 / step))
    min_units = max(0, int(round(min_weight / step)))
    if min_units * len(DIST_FEATURES) > units:
        min_units = 0
    parts = [min_units] * len(DIST_FEATURES)
    remaining_units = units - min_units * len(DIST_FEATURES)
    out: List[Dict[str, float]] = []

    def fill(idx: int, remaining: int) -> None:
        if idx == len(DIST_FEATURES) - 1:
            parts[idx] = min_units + remaining
            out.append({feature: parts[i] / units for i, feature in enumerate(DIST_FEATURES)})
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

    payload = base_payload("distribution", len(candidates))
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "出货分权重实验：PIT 组合 P14/P16/P17/P19/P20/P22、近期龙虎榜、technical 和原始 divergence，检验未来超额收益是否随出货分升高而降低。",
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
        ("p16_only", {"p16": 1.0}),
        ("p17_only", {"p17": 1.0}),
        ("p19_only", {"p19": 1.0}),
        ("p20_only", {"p20": 1.0}),
        ("p22_only", {"p22": 1.0}),
        ("lhb_recent_only", {"lhb_recent": 1.0}),
        ("divergence_only", {"divergence": 1.0}),
        ("patterns_equal", {feature: 1 / 6 for feature in ("p14", "p16", "p17", "p19", "p20", "p22")}),
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
        "selection_rule": "线上出货分使用人工指定权重；九项特征均保留，单因子/分组消融仅作为报告参考。",
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


# ── accumulation：七原始分吸筹总分权重实验 ─────────────────────

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


def _collect_accumulation_samples(conn: sqlite3.Connection,
                                  candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PIT 收集七个吸筹原始特征 + 未来超额收益。"""
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
            fired = match_patterns(code, window)
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
    units = int(round(1.0 / step))
    min_units = max(0, int(round(min_weight / step)))
    if min_units * len(ACCUM_FEATURES) > units:
        min_units = 0
    parts = [min_units] * len(ACCUM_FEATURES)
    remaining_units = units - min_units * len(ACCUM_FEATURES)
    out: List[Dict[str, float]] = []

    def fill(idx: int, remaining: int) -> None:
        if idx == len(ACCUM_FEATURES) - 1:
            parts[idx] = min_units + remaining
            out.append({feature: parts[i] / units for i, feature in enumerate(ACCUM_FEATURES)})
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
        collected = _collect_accumulation_samples(conn, candidates)
    finally:
        conn.close()

    samples = collected["samples"]
    dates = collected["dates"]
    train_dates, val_dates = _distribution_date_split(dates)
    train_groups = _prepare_accumulation_groups(samples, train_dates)
    val_groups = _prepare_accumulation_groups(samples, val_dates)
    all_groups = _prepare_accumulation_groups(samples, dates)

    payload = base_payload("accumulation", len(candidates))
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "吸筹总分权重实验：用 chip、position、cmf_eff、P3、P24、股东户数变化、公司回购七个原始分直接加权，不用截面 rank 合成线上分数。",
        "params": {
            "horizons": list(ACCUM_EXPERIMENT_HORIZONS),
            "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS,
            "min_names_per_section": VERIFY_MIN_NAMES,
            "grid_step": ACCUM_EXPERIMENT_GRID_STEP,
            "min_weight": ACCUM_EXPERIMENT_MIN_WEIGHT,
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

    market_structure_equal = {
        k: (1 / 5 if k in ("chip", "position", "cmf_eff", "p3", "p24") else 0.0)
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
            "selection_rule": "线上推荐只从七项权重均>=0.1的网格候选中选择；单因子/分组消融仅作为报告参考，不参与推荐。",
            "validation_candidate_count": len(candidates_grid),
            "top_models": top_models,
            "validation_top_models": validation_top,
            "ablations": ablations,
            "selected_model": ablations[-2],
            "recommended_model": recommended,
        },
        "notes": [
            "线上吸筹总分使用 chip、position、cmf_eff、P3、P24、股东户数变化、公司回购七个原始分按权重相加。",
            "实验里的 RankIC 仅用于评价分数排序能力；高低差是高分五分位相对低分五分位的未来超额收益差。",
        ],
    })
    write_payload(ACCUM_EXPERIMENT_FILE, payload)
    _print_accumulation_summary(payload)
    return payload


def _print_accumulation_summary(payload: Dict[str, Any]) -> None:
    print("=" * 112)
    print("  主力资金雷达 · 七原始分吸筹总分权重实验 (accumulation)")
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
    payload = base_payload("watch", len(candidates))
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

    payload = base_payload("latent", len(rows))
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
            "research": "meta_data_backup/hot_money_latent_hunting_research.md",
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
             "distribution(出货分权重实验) / accumulation(七原始分吸筹总分权重实验) / "
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
        help="候选池：leader=细分龙头(默认) / hotmoney=游资小盘universe(近1年龙虎榜≥5次·流通市值≤100亿)。",
    )
    return parser


def run_mode(mode: str, as_of: Optional[str] = None,
             max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    if mode == "watch":
        return run_watch(max_cap=max_cap, pool=pool)
    if mode == "verify":
        return run_verify(max_cap=max_cap, pool=pool)
    if mode == "patterns":
        return run_patterns(max_cap=max_cap, pool=pool)
    if mode == "distribution":
        return run_distribution_experiment(max_cap=max_cap, pool=pool)
    if mode == "accumulation":
        return run_accumulation_experiment(max_cap=max_cap, pool=pool)
    if mode == "latent":
        return run_latent(as_of=as_of, max_cap=max_cap, pool=pool)
    return run_ambush(as_of=as_of, max_cap=max_cap, pool=pool)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    # hotmoney 池建池时已按流通市值≤100亿过滤，不再按总市值剔除；市值过滤仅对 leader 池生效
    max_cap = MAX_MARKET_CAP_YI if (args.exclude_large_cap and args.pool == "leader") else None
    return run_mode(args.mode, as_of=args.as_of, max_cap=max_cap, pool=args.pool)


if __name__ == "__main__":
    main()
