CREATE DATABASE IF NOT EXISTS RETAIL_DW;
USE DATABASE RETAIL_DW;
CREATE SCHEMA IF NOT EXISTS POS_DATA;
CREATE SCHEMA IF NOT EXISTS FEATURES;
CREATE SCHEMA IF NOT EXISTS PREDICTIONS;

CREATE TABLE IF NOT EXISTS POS_DATA.TRANSACTIONS (
    transaction_id  VARCHAR(36) PRIMARY KEY,
    location_id     VARCHAR(20) NOT NULL,
    sku             VARCHAR(50) NOT NULL,
    quantity        INTEGER NOT NULL,
    unit_price      DECIMAL(10,2) NOT NULL,
    transaction_ts  TIMESTAMP_NTZ NOT NULL,
    channel         VARCHAR(20),
    region          VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS POS_DATA.INVENTORY (
    snapshot_ts      TIMESTAMP_NTZ NOT NULL,
    location_id      VARCHAR(20) NOT NULL,
    sku              VARCHAR(50) NOT NULL,
    quantity_on_hand INTEGER NOT NULL,
    reorder_point    INTEGER,
    PRIMARY KEY (snapshot_ts, location_id, sku)
);

CREATE TABLE IF NOT EXISTS FEATURES.DEMAND_FEATURES (
    feature_date     DATE NOT NULL,
    location_id      VARCHAR(20) NOT NULL,
    sku              VARCHAR(50) NOT NULL,
    units_sold_7d    DECIMAL(12,2),
    units_sold_14d   DECIMAL(12,2),
    units_sold_28d   DECIMAL(12,2),
    units_sold_90d   DECIMAL(12,2),
    avg_price_7d     DECIMAL(10,2),
    seasonality_idx  DECIMAL(5,4),
    promotion_active BOOLEAN,
    PRIMARY KEY (feature_date, location_id, sku)
);

CREATE TABLE IF NOT EXISTS PREDICTIONS.DEMAND_FORECAST (
    prediction_ts   TIMESTAMP_NTZ NOT NULL,
    location_id     VARCHAR(20) NOT NULL,
    sku             VARCHAR(50) NOT NULL,
    forecast_date   DATE NOT NULL,
    predicted_units DECIMAL(10,2) NOT NULL,
    conf_lower      DECIMAL(10,2),
    conf_upper      DECIMAL(10,2),
    model_version   VARCHAR(20)
);
