-- 1. 데이터베이스 생성
-- 'minwon_db'라는 이름의 데이터베이스가 없으면 새로 생성합니다.
-- 문자 인코딩을 utf8mb4로 설정하여 한글과 이모지를 문제없이 지원하도록 합니다.
CREATE DATABASE IF NOT EXISTS `minwon_db`
DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 2. 사용자 생성 및 권한 부여
-- 'localhost'에서 접속하는 'minwon_user' 사용자를 생성하고 비밀번호를 '1234'로 설정합니다.
-- 주의: 실제 운영 환경에서는 더 강력한 비밀번호를 사용해야 합니다.
CREATE USER IF NOT EXISTS 'minwon_user'@'localhost' IDENTIFIED BY '1234';

-- 'minwon_db' 데이터베이스의 모든 테이블에 대한 모든 권한을 'minwon_user'에게 부여합니다.
GRANT ALL PRIVILEGES ON `minwon_db`.* TO 'minwon_user'@'localhost';

-- 변경된 권한을 즉시 시스템에 적용합니다.
FLUSH PRIVILEGES;

