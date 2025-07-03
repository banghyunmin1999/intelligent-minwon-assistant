-- ====================================================================
-- 1. 데이터베이스(스키마) 생성
-- ====================================================================
DROP SCHEMA IF EXISTS `minwon_gov`;
CREATE SCHEMA `minwon_gov` DEFAULT CHARACTER SET utf8mb4;


-- ====================================================================
-- 2. 데이터베이스 사용자 생성 및 권한 부여 (인증 방식 수정)
-- ====================================================================
-- 만약 사용자가 이미 존재하면 삭제하여 충돌을 방지합니다.
DROP USER IF EXISTS 'gov_user'@'%';
FLUSH PRIVILEGES;

-- ▼▼▼▼▼ 여기가 수정된 부분입니다 ▼▼▼▼▼
-- 이전 버전과 호환되는 mysql_native_password 인증 방식을 사용하여 사용자를 생성합니다.
CREATE USER 'gov_user'@'%' IDENTIFIED WITH mysql_native_password BY '1234';
-- ▲▲▲▲▲ 여기가 수정된 부분입니다 ▲▲▲▲▲

-- 'gov_user' 사용자에게 'minwon_gov' 데이터베이스에 대한 모든 권한을 부여합니다.
GRANT ALL PRIVILEGES ON minwon_gov.* TO 'gov_user'@'%';
FLUSH PRIVILEGES;


-- ====================================================================
-- 3. 테이블 생성 (minwon_gov 데이터베이스 사용)
-- ====================================================================
USE `minwon_gov`;

-- 사용자 정보 테이블
CREATE TABLE `TBL_USERS` (
  `user_id` VARCHAR(50) NOT NULL,
  `user_pw` VARCHAR(255) NOT NULL,
  `user_name` VARCHAR(100) NOT NULL,
  `user_role` VARCHAR(20) NOT NULL DEFAULT 'USER',
  `reg_date` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 민원 정보 테이블
CREATE TABLE `TBL_COMPLAINTS` (
  `complaint_id` INT NOT NULL AUTO_INCREMENT,
  `title` VARCHAR(255) NOT NULL,
  `content` TEXT NOT NULL,
  `file_path` VARCHAR(500) NULL,
  `status` VARCHAR(20) NOT NULL DEFAULT 'RECEIVED',
  `submitter_id` VARCHAR(50) NOT NULL,
  `assignee_id` VARCHAR(50) NULL,
  `submit_date` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `completed_date` TIMESTAMP NULL,
  PRIMARY KEY (`complaint_id`),
  CONSTRAINT `fk_complaints_to_users` FOREIGN KEY (`submitter_id`) REFERENCES `TBL_USERS` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 답변 정보 테이블
CREATE TABLE `TBL_ANSWERS` (
  `answer_id` INT NOT NULL AUTO_INCREMENT,
  `complaint_id` INT NOT NULL,
  `content` TEXT NOT NULL,
  `is_final` TINYINT(1) NOT NULL DEFAULT 1,
  `writer_id` VARCHAR(50) NOT NULL,
  `write_date` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`answer_id`),
  CONSTRAINT `fk_answers_to_complaints` FOREIGN KEY (`complaint_id`) REFERENCES `TBL_COMPLAINTS` (`complaint_id`),
  CONSTRAINT `fk_answers_to_users` FOREIGN KEY (`writer_id`) REFERENCES `TBL_USERS` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


UPDATE TBL_USERS SET user_role = 'ADMIN' WHERE user_id = 'admin';