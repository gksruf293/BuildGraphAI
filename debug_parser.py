import json

def analyze_llm_clustering(req_file, res_file):
    # 1. JSON 파일 로드
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            req_data = json.load(f)
        with open(res_file, 'r', encoding='utf-8') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        print("디버그 파일을 찾을 수 없습니다.")
        return

    # 2. ID를 실제 데이터로 매핑하기 위한 딕셔너리 생성
    text_map = {t['id']: t['text'] for t in req_data['texts']}
    
    print("\n🧠 LLM 시맨틱 클러스터링 결과 분석")
    print("=" * 50)

    # 3. 방(Room)별로 할당된 결과 해석
    assigned_fixture_set = set()
    
    for room in res_data['rooms']:
        room_name = room.get('room_name', 'UNKNOWN')
        
        # 할당된 텍스트 ID를 실제 텍스트로 변환
        assigned_texts = [text_map[t_id] for t_id in room.get('assigned_text_ids', []) if t_id in text_map]
        
        # 할당된 가구 개수 파악
        assigned_fixtures = room.get('assigned_fixture_ids', [])
        assigned_fixture_set.update(assigned_fixtures)
        
        print(f"🏠 [방 구획]: {room_name}")
        print(f"   🏷️ 묶인 라벨들: {', '.join(assigned_texts)}")
        print(f"   🪑 할당된 가구 수: {len(assigned_fixtures)}개 (ID: {', '.join(assigned_fixtures[:3])}...)\n")

    # 4. 누락된 데이터 확인 (디버깅 핵심 포인트)
    total_fixtures = len(req_data['fixtures'])
    missing_fixtures = total_fixtures - len(assigned_fixture_set)
    
    print("⚠️ [데이터 누락 점검]")
    print("-" * 50)
    print(f"   - LLM에 보낸 총 가구 선분: {total_fixtures}개")
    print(f"   - LLM이 방에 할당한 선분: {len(assigned_fixture_set)}개")
    print(f"   - LLM이 무시한 선분: {missing_fixtures}개")
    
    if missing_fixtures > 0:
        print("\n🚨 [진단 결과]: LLM이 위치(좌표) 정보 없이 수많은 'FIXTURE' 텍스트만 보고")
        print("할당을 포기하거나 환각(Hallucination)을 일으켰습니다.")

# 실행
analyze_llm_clustering('debug_llm_request.json', 'debug_llm_response.json')