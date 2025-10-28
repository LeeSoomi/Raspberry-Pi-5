# sudo apt-get update
# sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
# sudo apt install -y python3-opencv python3-pandas python3-numpy python3-pytesseract
# >> 확인 
# python3 -m pip show opencv-python
# python3 -m pip show pytesseract

# 카메라 권한 필요
# sudo usermod -a -G video $USER

# 설치확인
# python3 -c "import cv2, pytesseract, numpy; print('OK')"

# # 1. Tesseract OCR 설치 (한글 포함)
# sudo apt-get install -y tesseract-ocr tesseract-ocr-kor

# # 2. Python 연동 패키지 설치
# pip3 install pytesseract numpy

# 설치 확인:
# bash# Tesseract 설치 확인
# tesseract --version

# # Python에서 확인
# python3 -c "import pytesseract; print('pytesseract OK')"



#!/usr/bin/env python3
"""
CSV 기반 4자리 번호 매칭 시스템
라즈베리파이5용
"""
# CSV 기반 4자리 매칭 (추천)
# python3 csv_based_matcher.py
import cv2
import numpy as np
import pytesseract
import re
import csv
import os
from datetime import datetime

class CSVBasedMatcher:
    def __init__(self, csv_file='allowed_vehicles.csv'):
        """CSV 기반 매칭 시스템"""
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        self.csv_file = csv_file
        self.allowed_list = self.load_csv()
        
        # 매칭 성공 카운터
        self.match_count = {}
        
    def load_csv(self):
        """CSV 파일에서 허가 목록 로드"""
        allowed = {}
        
        if not os.path.exists(self.csv_file):
            # 샘플 CSV 생성
            print(f"CSV 파일이 없습니다. 샘플 파일 생성: {self.csv_file}")
            self.create_sample_csv()
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    allowed[row['last_4digits']] = {
                        'name': row['name'],
                        'vehicle': row['vehicle'],
                        'type': row['type']
                    }
            print(f"CSV 로드 완료: {len(allowed)}개 항목")
        except Exception as e:
            print(f"CSV 로드 오류: {e}")
            
        return allowed
    
    def create_sample_csv(self):
        """샘플 CSV 파일 생성"""
        headers = ['last_4digits', 'name', 'vehicle', 'type']
        sample_data = [
            ['1234', '홍길동', '소나타', '직원'],
            ['5678', '김철수', '그랜저', 'VIP'],
            ['9012', '이영희', '아반떼', '방문자'],
            ['3456', '박민수', 'K5', '직원'],
            ['7890', '최영수', '투싼', '거주자']
        ]
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(sample_data)
    
    def extract_4digits_multi_method(self, image):
        """여러 방법으로 4자리 추출 시도"""
        results = []
        
        # 방법 1: 기본 OCR
        text1 = pytesseract.image_to_string(image, config='--psm 8')
        digits1 = re.findall(r'\d{4,}', text1)
        if digits1:
            results.extend([d[-4:] for d in digits1])
        
        # 방법 2: 숫자만 인식
        text2 = pytesseract.image_to_string(
            image, 
            config='--psm 11 -c tessedit_char_whitelist=0123456789'
        )
        digits2 = re.findall(r'\d{4,}', text2)
        if digits2:
            results.extend([d[-4:] for d in digits2])
        
        # 방법 3: 전처리 후 인식
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text3 = pytesseract.image_to_string(binary, config='--psm 7')
        digits3 = re.findall(r'\d{4,}', text3)
        if digits3:
            results.extend([d[-4:] for d in digits3])
        
        # 중복 제거 및 가장 빈번한 결과 반환
        if results:
            from collections import Counter
            counter = Counter(results)
            most_common = counter.most_common(1)[0][0]
            return most_common[-4:]  # 마지막 4자리만
        
        return None
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 그레이스케일
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 크기 조정
        height, width = gray.shape
        if width < 200:
            scale = 200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        return gray
    
    def find_number_regions(self, frame):
        """숫자가 있을 만한 영역 찾기"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 모폴로지 연산으로 숫자 영역 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 크기 필터
            if w > 50 and h > 15 and w < frame.shape[1] * 0.8:
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 6.0:
                    # 여백 추가
                    margin = 5
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + 2 * margin)
                    h = min(frame.shape[0] - y, h + 2 * margin)
                    
                    regions.append((x, y, w, h))
        
        return regions
    
    def log_access(self, digits, allowed, info=None):
        """접근 기록 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open('access_log.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if allowed and info:
                writer.writerow([timestamp, digits, info['name'], info['vehicle'], '승인'])
                print(f"[승인] {timestamp} - {digits} ({info['name']})")
            else:
                writer.writerow([timestamp, digits, '-', '-', '거부'])
                print(f"[거부] {timestamp} - {digits} (미등록)")
    
    def process_video_frame(self, frame):
        """비디오 프레임 처리"""
        results = []
        
        # 숫자 영역 찾기
        regions = self.find_number_regions(frame)
        
        for x, y, w, h in regions:
            roi = frame[y:y+h, x:x+w]
            
            # 전처리
            preprocessed = self.preprocess_image(roi)
            
            # 4자리 추출
            four_digits = self.extract_4digits_multi_method(preprocessed)
            
            if four_digits and four_digits.isdigit() and len(four_digits) == 4:
                # 매칭 확인
                if four_digits in self.allowed_list:
                    info = self.allowed_list[four_digits]
                    results.append({
                        'bbox': (x, y, w, h),
                        'digits': four_digits,
                        'allowed': True,
                        'info': info
                    })
                else:
                    results.append({
                        'bbox': (x, y, w, h),
                        'digits': four_digits,
                        'allowed': False,
                        'info': None
                    })
        
        return results
    
    def draw_results(self, frame, results):
        """결과 시각화"""
        for result in results:
            x, y, w, h = result['bbox']
            
            if result['allowed']:
                color = (0, 255, 0)  # 녹색
                info = result['info']
                label = f"{result['digits']} - {info['name']} ({info['type']})"
            else:
                color = (0, 0, 255)  # 빨간색
                label = f"{result['digits']} - 미등록"
            
            # 박스와 라벨
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # 배경 박스
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y-20), (x+label_size[0], y), color, -1)
            
            # 텍스트
            cv2.putText(frame, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


def main():
    print("=== CSV 기반 4자리 매칭 시스템 ===")
    
    matcher = CSVBasedMatcher()
    
    # 로그 파일 초기화
    if not os.path.exists('access_log.csv'):
        with open('access_log.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['시간', '4자리', '이름', '차량', '상태'])
    
    print("\n1. 실시간 모니터링")
    print("2. CSV 파일 편집")
    print("3. 접근 로그 확인")
    
    choice = input("선택: ")
    
    if choice == '1':
        # 실시간 모니터링
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n실시간 모니터링 시작...")
        print("종료: q | CSV 재로드: r")
        
        # 중복 감지 방지용
        last_detected = {}
        detection_cooldown = 5  # 5초
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 처리
            results = matcher.process_video_frame(frame)
            
            # 새로운 감지만 로그
            current_time = datetime.now().timestamp()
            for result in results:
                digits = result['digits']
                
                if digits not in last_detected or \
                   current_time - last_detected[digits] > detection_cooldown:
                    matcher.log_access(digits, result['allowed'], result.get('info'))
                    last_detected[digits] = current_time
            
            # 결과 표시
            frame = matcher.draw_results(frame, results)
            
            # 상태 표시
            cv2.putText(frame, f"등록: {len(matcher.allowed_list)}명", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('CSV 4-Digit Matcher', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # CSV 재로드
                matcher.allowed_list = matcher.load_csv()
                print("CSV 파일 재로드 완료")
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif choice == '2':
        # CSV 편집 안내
        print(f"\nCSV 파일 위치: {os.path.abspath(matcher.csv_file)}")
        print("엑셀이나 텍스트 에디터로 편집하세요.")
        print("\n형식:")
        print("last_4digits,name,vehicle,type")
        print("1234,홍길동,소나타,직원")
        
    elif choice == '3':
        # 로그 확인
        print("\n=== 최근 접근 로그 ===")
        try:
            with open('access_log.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # 최근 20개만 표시
                for row in rows[-20:]:
                    print(' | '.join(row))
        except:
            print("로그 파일이 없습니다.")


if __name__ == "__main__":
    main()
