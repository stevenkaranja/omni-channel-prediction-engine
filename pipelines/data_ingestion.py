"""POS data ingestion: Kafka → Snowflake."""
import json
import logging
from kafka import KafkaConsumer
import snowflake.connector
import yaml

logger = logging.getLogger(__name__)

class POSIngestionPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.sf = self._connect_snowflake()
        self.consumer = self._create_consumer()

    def _connect_snowflake(self):
        import os
        return snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            database=self.cfg["snowflake"]["database"],
            warehouse=self.cfg["snowflake"]["warehouse"],
            schema=self.cfg["snowflake"]["schema"],
        )

    def _create_consumer(self):
        import os
        return KafkaConsumer(
            "pos.transactions",
            bootstrap_servers=os.environ["KAFKA_BROKERS"].split(","),
            value_deserializer=lambda v: json.loads(v.decode()),
            group_id="omni-channel-ingestion",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )

    def run(self, batch_size: int = 500):
        batch = []
        cursor = self.sf.cursor()
        for msg in self.consumer:
            batch.append(msg.value)
            if len(batch) >= batch_size:
                self._flush(cursor, batch)
                batch.clear()

    def _flush(self, cursor, batch: list):
        rows = [
            (r["transaction_id"], r["location_id"], r["sku"],
             r["quantity"], r["unit_price"], r["transaction_ts"],
             r.get("channel"), r.get("region"))
            for r in batch
        ]
        cursor.executemany(
            "INSERT INTO POS_DATA.TRANSACTIONS VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            rows
        )
        logger.info(f"Flushed {len(rows)} rows to Snowflake")
