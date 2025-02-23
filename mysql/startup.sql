CREATE USER stocks@'%' IDENTIFIED BY '1234';
GRANT ALL PRIVILEGES ON stocks.* TO stocks;

CREATE USER grafana@'%' IDENTIFIED BY '1234';
GRANT ALL PRIVILEGES ON stocks.* TO grafana;

USE stocks;

CREATE TABLE `stocks`.`price_history` (
    `id` bigint auto_increment primary key,
    `ticker` varchar(10) not null,
    `reference_date` date not null default (current_date),
    `close_price` double not null,
    CONSTRAINT `unk_ticker_date` UNIQUE (`ticker`, `reference_date`),
    INDEX `idx_price_dt` (`reference_date`)
);

CREATE TABLE `stocks`.`prediction_price_history` (
    `id` bigint auto_increment primary key,
    `ticker` varchar(10) not null,
    `reference_date` date not null default (current_date),
    `predicted_price` double not null,
    `model_version` varchar(255) not null,
    CONSTRAINT `unk_ticker_date+prediction` UNIQUE (`ticker`, `reference_date`),
    INDEX `idx_prediction_price_dt` (`reference_date`),
    INDEX `idx_prediction_ticker` (`ticker`),
    INDEX `idx_prediction_model_version` (`model_version`)
);