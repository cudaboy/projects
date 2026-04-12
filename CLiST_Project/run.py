import argparse
import json
from clist.pipeline import CLiSTPipeline

def main():
    # 1. 터미널 명령어 파서(Parser) 생성
    parser = argparse.ArgumentParser(description="CLiST 팩토리 예지보전 멀티모달 AI 시스템")
    
    # 2. 명령어 옵션 정의
    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'train', 'tune'], 
                        help="실행 모드를 선택하세요 (predict, train, tune)")
    parser.add_argument('--sensor', type=str, help="[predict 모드] 분석할 센서 CSV 파일 경로")
    parser.add_argument('--vision', type=str, help="[predict 모드] 분석할 열화상 BIN 파일 경로")
    parser.add_argument('--weight', type=str, default='weights/best_clist_model.pth', 
                        help="가중치 파일 경로 (기본값: weights/best_clist_model.pth)")

    args = parser.parse_args()

    # 3. 모드별 라우팅 (Routing)
    if args.mode == 'predict':
        if not args.sensor or not args.vision:
            print("❌ 에러: 'predict' 모드에서는 --sensor와 --vision 파일 경로를 모두 입력해야 합니다.")
            return
            
        print("🔍 모델 및 가중치를 로드하는 중...")
        pipeline = CLiSTPipeline(weight_path=args.weight)
        
        print(f"📡 데이터 분석 중... \n - 센서: {args.sensor}\n - 열화상: {args.vision}")
        result = pipeline.predict(sensor_csv_path=args.sensor, vision_bin_path=args.vision)
        
        print("\n" + "="*50)
        print(f"🚨 [최종 판정 결과]: {result['predicted_status']} (신뢰도: {result['confidence']*100:.1f}%)")
        print("="*50)
        print("📊 [클래스별 확률 분포]")
        print(json.dumps(result['all_probabilities'], indent=4, ensure_ascii=False))
        
    elif args.mode == 'train':
        print("🚀 학습 스크립트(scripts/train.py)를 실행합니다...")
        import scripts.train as train
        train.main()
        
    elif args.mode == 'tune':
        print("🚀 하이퍼파라미터 튜닝 스크립트(scripts/tune.py)를 실행합니다...")
        # tune.py는 내부적으로 os.system 등으로 호출하도록 구성하거나 직접 import 처리
        import scripts.tune as tune
        tune.main()

if __name__ == "__main__":
    main()