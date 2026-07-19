# -*- coding: utf-8 -*-
import scrapy
import os
import json

from lxml import etree
from fund.ttjj_spider.items import MyItem


def _first(values, default=''):
    return values[0].strip() if values else default


def _join_text(values):
    return ''.join(v.strip() for v in values if v is not None).strip()


def _extract_five_year_growth(text):
    try:
        return text.split("近5年")[1].split("成立来")[0].split("</li>")[1].split(">")[1].strip()
    except (IndexError, ValueError):
        return ''


class JijinSpider(scrapy.Spider):
    name = 'jijin'
    allowed_domains = ['fund.eastmoney.com', 'fundf10.eastmoney.com']

    data_dir = os.path.join('.', 'data')
    fund_codes_file = os.path.join(data_dir, 'fund_codes.json')

    handle_httpstatus_list = [404, 500]

    def start_requests(self):
        try:
            with open(self.fund_codes_file, 'r', encoding='utf-8') as fin:
                fund_codes = json.load(fin)
        except FileNotFoundError:
            self.logger.error('%s 不存在，请先通过基金编辑器保存配置或运行 python fund_data_refresh.py', self.fund_codes_file)
            return
        except json.JSONDecodeError as exc:
            self.logger.error('%s 不是合法 JSON: %s', self.fund_codes_file, exc)
            return

        for code in fund_codes:
            yield scrapy.Request(
                'http://fund.eastmoney.com/{}.html'.format(code),
                callback=self.parse,
                meta={'fund_code': code},
            )

    def parse(self, response):
        if response.status != 200:
            self.logger.warning('基金页面请求失败: %s status=%s', response.url, response.status)
            return

        my_item = MyItem()
        html = etree.HTML(response.text)
        if html is None:
            self.logger.warning('基金页面解析失败: %s', response.url)
            return

        name = _first(html.xpath('//div[@class="fundDetail-tit"]/div[1]/text()'))
        fund_code = _first(html.xpath('//*[@id="body"]/div[11]/div/div/div[1]/div[1]/div//span[@class="ui-num"]/text()'))
        manager_trigger = _first(html.xpath('//*[@id="fundManagerTab"]//td[@class="td03"]/text()'))
        fund_info_class = _first(html.xpath('//div[@class="fundInfoItem"]/div[1]/@class'))

        if not fund_code:
            fund_code = response.meta.get('fund_code', '')
        if not name:
            self.logger.warning('跳过基金 %s: 名称缺失', fund_code or response.url)
            return
        if fund_info_class == 'tuishiTip':
            self.logger.warning('跳过基金 %s %s: 已退市/异常状态', fund_code, name)
            return

        my_item['name'] = name
        my_item['fundCode'] = fund_code
        my_item['managerTrigger'] = manager_trigger
        my_item['netAssetValueEstimated'] = _join_text(html.xpath('//dl[@class="dataItem01"]/dd[1]/dl[1]/span/text()'))
        my_item['netAssetValue'] = _join_text(html.xpath('//dl[@class="dataItem02"]/dd[1]/span[1]/text()'))
        my_item['netAssetValueAccumulated'] = _join_text(html.xpath('//dl[@class="dataItem03"]/dd[1]/span/text()'))
        my_item['riskRating'] = _join_text(html.xpath('//div[@class="infoOfFund"]/table/tr[2]/td[3]/div/@class'))

        for row in html.xpath('//*[@id="increaseAmount_stage"]/table/tr[position()>1]'):
            title = _join_text(row.xpath('./td[1]/div/text()'))
            if title.startswith('阶段涨幅'):
                key = 'netAssetValueRestoredGrowthRate'
            elif title.startswith('同类平均'):
                key = 'categoryAverageOfNetAssetValueRestoredGrowth'
            elif title.startswith('沪深300'):
                key = 'hs300GrowthRate'
            elif title.startswith('同类排名'):
                key = 'rankInCategoryOfNetAssetValueRestoredGrowth'
            elif title.startswith('四分位排名'):
                key = 'quartileRankInCategoryOfNetValueRestoredGrowth'
            else:
                key = ''

            if key == 'quartileRankInCategoryOfNetValueRestoredGrowth':
                text_node = 'h3'
            elif key != '':
                text_node = 'div'
            else:
                continue

            my_item['{}RecentWeek'.format(key)] = _join_text(row.xpath(f'./td[2]/{text_node}/text()'))
            my_item['{}RecentMonth'.format(key)] = _join_text(row.xpath(f'./td[3]/{text_node}/text()'))
            my_item['{}RecentThreeMonth'.format(key)] = _join_text(row.xpath(f'./td[4]/{text_node}/text()'))
            my_item['{}RecentSixMonth'.format(key)] = _join_text(row.xpath(f'./td[5]/{text_node}/text()'))
            my_item['{}RecentOneYear'.format(key)] = _join_text(row.xpath(f'./td[7]/{text_node}/text()'))
            my_item['{}RecentTwoYear'.format(key)] = _join_text(row.xpath(f'./td[8]/{text_node}/text()'))
            my_item['{}RecentThreeYear'.format(key)] = _join_text(row.xpath(f'./td[9]/{text_node}/text()'))
            my_item['{}SinceFirstDayOfYear'.format(key)] = _join_text(row.xpath(f'./td[6]/{text_node}/text()'))

        manager_urls = html.xpath('//*[@id="fundManagerTab"]//td[@class="td02"]//a/@href')[:1]
        manager_url = response.urljoin(manager_urls[0]) if manager_urls else ''
        zhangdie_url = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jdzf&code={}".format(fund_code)
        yield scrapy.Request(
            zhangdie_url,
            callback=self.parse_five_year_growth,
            errback=self.handle_five_year_error,
            meta={'item': my_item, 'manager_url': manager_url},
        )

    def parse_five_year_growth(self, response):
        item = response.meta['item']
        manager_url = response.meta.get('manager_url')

        content = _extract_five_year_growth(response.text) if response.status == 200 else ''
        if content:
            item['netAssetValueRestoredGrowthRateRecentFiveYear'] = content
        else:
            item['netAssetValueRestoredGrowthRateRecentFiveYear'] = ''
            self.logger.warning('基金 %s 近5年涨幅缺失: %s', item.get('fundCode'), response.url)

        yield from self._request_manager_or_yield(item, manager_url)

    def handle_five_year_error(self, failure):
        item = failure.request.meta['item']
        manager_url = failure.request.meta.get('manager_url')
        item['netAssetValueRestoredGrowthRateRecentFiveYear'] = ''
        self.logger.warning('基金 %s 近5年涨幅请求失败: %s', item.get('fundCode'), failure.value)
        yield from self._request_manager_or_yield(item, manager_url)

    def _request_manager_or_yield(self, item, manager_url):
        if manager_url:
            yield scrapy.Request(
                manager_url,
                callback=self.parse_manager,
                errback=self.handle_manager_error,
                dont_filter=True,
                meta={'item': item},
            )
        else:
            # 缺失置空，下游显示 '--'；不能用大数哨兵，否则会被当成真实规模染色
            item['fund_manager_total_asset'] = ''
            yield item

    def parse_manager(self, response):
        item = response.meta['item']
        html = etree.HTML(response.text)
        asset = ''
        if html is not None:
            asset = _first(html.xpath('/html/body/div[6]/div[2]/div[1]/div/div[2]/div[2]/div/div[1]/span[2]/span[1]/text()'))
        if not asset:
            self.logger.warning('基金 %s 基金经理管理规模缺失: %s', item.get('fundCode'), response.url)
        item['fund_manager_total_asset'] = asset
        yield item

    def handle_manager_error(self, failure):
        item = failure.request.meta['item']
        self.logger.warning('基金 %s 基金经理管理规模请求失败: %s', item.get('fundCode'), failure.value)
        item['fund_manager_total_asset'] = ''
        yield item
