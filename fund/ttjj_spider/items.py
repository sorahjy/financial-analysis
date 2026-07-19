# -*- coding: utf-8 -*-

# Define here the models for fund profile items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MyItem(scrapy.Item):
    body = scrapy.Field()

    # 主页面
    name = scrapy.Field()  # 名称
    fund_manager_total_asset = scrapy.Field()
    fundCode = scrapy.Field()
    managerTrigger = scrapy.Field()
    netAssetValueEstimated = scrapy.Field()  # 估算净值
    netAssetValue = scrapy.Field()  # 单位净值
    netAssetValueAccumulated = scrapy.Field()  # 累计单位净值
    riskRating = scrapy.Field()  # 评级
    netAssetValueRestoredGrowthRateRecentWeek = scrapy.Field()  # 近一周涨幅
    netAssetValueRestoredGrowthRateRecentMonth = scrapy.Field()  # 近一月涨幅
    netAssetValueRestoredGrowthRateRecentThreeMonth = scrapy.Field()  # 近三月涨幅
    netAssetValueRestoredGrowthRateRecentSixMonth = scrapy.Field()  # 近六月涨幅
    netAssetValueRestoredGrowthRateRecentOneYear = scrapy.Field()  # 近一年涨幅
    netAssetValueRestoredGrowthRateRecentTwoYear = scrapy.Field()  # 近两年涨幅
    netAssetValueRestoredGrowthRateRecentThreeYear = scrapy.Field()  # 近三年涨幅
    netAssetValueRestoredGrowthRateRecentFiveYear = scrapy.Field()  # 近五年涨幅
    netAssetValueRestoredGrowthRateSinceFirstDayOfYear = scrapy.Field()  # 今年来涨幅
    # netAssetValueRestoredGrowthRateSinceInception = scrapy.Field()  # 成立来涨幅
    categoryAverageOfNetAssetValueRestoredGrowthRecentWeek = scrapy.Field()  # 近一周同类平均
    categoryAverageOfNetAssetValueRestoredGrowthRecentMonth = scrapy.Field()  # 近一月同类平均
    categoryAverageOfNetAssetValueRestoredGrowthRecentThreeMonth = scrapy.Field()  # 近三月同类平均
    categoryAverageOfNetAssetValueRestoredGrowthRecentSixMonth = scrapy.Field()  # 近六月同类平均
    categoryAverageOfNetAssetValueRestoredGrowthRecentOneYear = scrapy.Field()  # 近一年同类平均
    categoryAverageOfNetAssetValueRestoredGrowthRecentTwoYear = scrapy.Field()  # 近两年同类平均
    categoryAverageOfNetAssetValueRestoredGrowthRecentThreeYear = scrapy.Field()  # 近三年同类平均
    # categoryAverageOfNetAssetValueRestoredGrowthRecentFiveYear = scrapy.Field()  # 近五年同类平均
    categoryAverageOfNetAssetValueRestoredGrowthSinceFirstDayOfYear = scrapy.Field()  # 今年来同类平均
    # categoryAverageOfNetAssetValueRestoredGrowthSinceInception = scrapy.Field()  # 成立来同类平均
    rankInCategoryOfNetAssetValueRestoredGrowthRecentWeek = scrapy.Field()  # 近一周同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthRecentMonth = scrapy.Field()  # 近一月同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthRecentThreeMonth = scrapy.Field()  # 近三月同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthRecentSixMonth = scrapy.Field()  # 近六月同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthRecentOneYear = scrapy.Field()  # 近一年同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthRecentTwoYear = scrapy.Field()  # 近两年同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthRecentThreeYear = scrapy.Field()  # 近三年同类排行
    # rankInCategoryOfNetAssetValueRestoredGrowthRecentFiveYear = scrapy.Field()  # 近五年同类排行
    rankInCategoryOfNetAssetValueRestoredGrowthSinceFirstDayOfYear = scrapy.Field()  # 今年来同类排行
    # rankInCategoryOfNetAssetValueRestoredGrowthSinceInception = scrapy.Field()  # 成立来同类排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentWeek = scrapy.Field()  # 近一周分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentMonth = scrapy.Field()  # 近一月分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentThreeMonth = scrapy.Field()  # 近三月分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentSixMonth = scrapy.Field()  # 近六月分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentOneYear = scrapy.Field()  # 近一年分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentTwoYear = scrapy.Field()  # 近两年分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthRecentThreeYear = scrapy.Field()  # 近三年分位排行
    # quartileRankInCategoryOfNetValueRestoredGrowthRecentFiveYear = scrapy.Field()  # 近五年分位排行
    quartileRankInCategoryOfNetValueRestoredGrowthSinceFirstDayOfYear = scrapy.Field()  # 今年来分位排行
    # quartileRankInCategoryOfNetValueRestoredGrowthSinceInception = scrapy.Field()  # 成立来分位排行
    hs300GrowthRateRecentWeek = scrapy.Field()  # 近一周沪深300涨幅
    hs300GrowthRateRecentMonth = scrapy.Field()  # 近一月沪深300涨幅
    hs300GrowthRateRecentThreeMonth = scrapy.Field()  # 近三月沪深300涨幅
    hs300GrowthRateRecentSixMonth = scrapy.Field()  # 近六月沪深300涨幅
    hs300GrowthRateRecentOneYear = scrapy.Field()  # 近一年沪深300涨幅
    hs300GrowthRateRecentTwoYear = scrapy.Field()  # 近两年沪深300涨幅
    hs300GrowthRateRecentThreeYear = scrapy.Field()  # 近三年沪深300涨幅
    # hs300GrowthRateRecentFiveYear = scrapy.Field()  # 近五年沪深300涨幅
    hs300GrowthRateSinceFirstDayOfYear = scrapy.Field()  # 今年来沪深300涨幅
    # hs300GrowthRateSinceInception = scrapy.Field()  # 成立来沪深300涨幅

