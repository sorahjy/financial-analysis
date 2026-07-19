# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import os

from fund.fund_storage import connect as connect_fund_db, save_profile_snapshots
# from pymongo import MongoClient


# def connect_db(config_file):
#     with open(config_file, 'r', encoding='utf-8') as fin:
#         config = json.load(fin)
#
#     host = config.get('host', '127.0.0.1')
#     port = config.get('port', 27017)
#     db_name = config.get('db_name', 'temp')
#     db_user = config.get('db_user', '')
#     db_passwd = config.get('db_passwd', '')
#
#     client = MongoClient(host=host, port=port)
#     if db_user != '':
#         client.admin.authenticate(db_user, db_passwd, mechanism=config.get('mongo_auth_mech', 'SCRAM-SHA-1'))
#     db = client.get_database(db_name)
#
#     return db


class TtjjSpiderPipeline(object):
    """把成功抓到的基金概况合并进 SQLite 快照。

    部分基金请求失败时保留其上一次快照；只有明确确认本轮覆盖全部预期
    基金代码时才做全量替换，以便删除配置中已移除的基金。
    """

    def __init__(self):
        self.data_dir = os.path.join('.', 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.items = []
        self.item_count = 0

        # self.db_config_file = os.path.join(self.data_dir, 'db_config.json')
        # self.db = connect_db(self.db_config_file)

    def process_item(self, item, spider):
        data = dict(item)
        self.items.append(data)
        self.item_count += 1

        # if self.db.jijin.find_one(data['fundCode']) is None:
        #     self.db.jijin.insert_one(data)

        return item

    def close_spider(self, spider):
        if self.item_count > 0:
            expected_codes = self._load_expected_codes(spider)
            actual_codes = {
                str(item.get('fundCode'))
                for item in self.items
                if item.get('fundCode')
            }
            complete = expected_codes is not None and actual_codes == expected_codes
            conn = connect_fund_db()
            try:
                save_profile_snapshots(
                    conn,
                    self.items,
                    replace=complete,
                    expected_codes=expected_codes if complete else None,
                )
            finally:
                conn.close()
            if complete:
                spider.logger.info('SQLite 基金概况完整快照已替换，共 %d 条', self.item_count)
            else:
                missing_count = len(expected_codes - actual_codes) if expected_codes is not None else 0
                spider.logger.warning(
                    'SQLite 基金概况仅合并成功结果，共 %d 条，保留旧数据%s',
                    self.item_count,
                    '，缺少 {} 只'.format(missing_count) if missing_count else '',
                )
        else:
            spider.logger.error('本次未爬到任何基金数据，保留旧 SQLite 基金概况快照')

    @staticmethod
    def _load_expected_codes(spider):
        codes_file = getattr(spider, 'fund_codes_file', None)
        if not codes_file:
            return None
        try:
            with open(codes_file, 'r', encoding='utf-8') as fin:
                payload = json.load(fin)
        except (OSError, ValueError, TypeError) as exc:
            spider.logger.warning('无法校验基金概况完整性，按增量合并处理: %s', exc)
            return None
        if not isinstance(payload, list):
            spider.logger.warning('基金代码配置不是列表，按增量合并处理')
            return None
        return {str(code) for code in payload}
