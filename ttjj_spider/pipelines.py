# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os

from fund_storage import connect as connect_fund_db, save_profile_snapshots
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
    """爬完且有数据才替换 SQLite 基金概况快照。

    避免爬虫启动即清空旧数据：一旦本次爬取整体失败，下游报告继续使用
    上一次的正常快照。
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
            conn = connect_fund_db()
            try:
                save_profile_snapshots(conn, self.items, replace=True)
            finally:
                conn.close()
            spider.logger.info('SQLite 基金概况快照已更新，共 %d 条', self.item_count)
        else:
            spider.logger.error('本次未爬到任何基金数据，保留旧 SQLite 基金概况快照')
