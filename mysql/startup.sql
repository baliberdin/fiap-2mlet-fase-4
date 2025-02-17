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
    `price_history_id` bigint not null,
    `predicted_close_price` double not null,
    `prediction_model_version` varchar(100) not null,
    CONSTRAINT FOREIGN KEY `fk_prediction_price_history` (`price_history_id`) REFERENCES `stocks`.`price_history`(`id`),
    CONSTRAINT `unk_price_history` UNIQUE (`price_history_id`)
);