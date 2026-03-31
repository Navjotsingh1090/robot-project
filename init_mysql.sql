-- One-time MySQL setup. Run: mysql -u root -p < init_mysql.sql

CREATE DATABASE IF NOT EXISTS robot
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE robot;

CREATE TABLE IF NOT EXISTS users (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name VARCHAR(191) NOT NULL,
  encoding BLOB NOT NULL,
  face_file_hash VARCHAR(64) NULL COMMENT 'SHA-256 hex of enrolled face JPEG',
  PRIMARY KEY (id),
  KEY idx_users_name (name)
) ENGINE=InnoDB;
