# 🏗️ AI BuildGraph Precision Pro

**2D CAD 도면(DXF) 시맨틱 파싱 및 공간 지식 그래프(GraphRAG) 추출 엔진**

https://buildgraphai-cjpdngwlnv4gvu2frjbthh.streamlit.app/

본 프로젝트는 복잡한 건축 CAD 도면(.dxf)을 업로드받아, AI(LLM) 기반의 시맨틱 텍스트 분석과 기하학적(Geometry) 연산을 통해 도면 내 공간(Room)을 자동으로 분할합니다. 분할된 공간은 면적, 축척, 인접성(Connectivity) 정보가 포함된 **지식 그래프(Knowledge Graph) JSON 데이터**로 변환되어, 다중 에이전트 논리 추론(예: Mafia Engine)의 고정밀 공간 환경 데이터로 활용됩니다.

## ✨ 주요 기능 (Key Features)
1. **LLM 시맨틱 분류 및 Fallback 엔진**: 도면 내 텍스트를 `ROOM`, `FIXTURE`, `DIMENSION`으로 지능형 분류하며, API 장애 시 자체 정규식 룰셋으로 전환되는 무장애(Zero-downtime) 파이프라인 적용.
2. **Arc-to-L-Shape 변환 (면적 보존)**: 문의 1/4원 궤적(Arc)을 기하학적으로 감지하여, 방의 코너 면적이 깎이지 않도록 중심점(Hinge)을 경유하는 직각(L자) 장벽으로 자동 변환.
3. **Zero-Threshold 노이즈 격리**: 도면상의 미세한 틈새나 벽체 내부 빈 공간(Cavity)을 억지로 방에 합치지 않고, `UNKNOWN_SPACE`로 완벽히 격리하여 노드(Node) 데이터의 순도를 100%로 유지.
4. **동적 캘리브레이션 (Dynamic Scale)**: 도면에 표기된 치수(Explicit)와 계산된 기하학적 치수(Estimated)를 비교하여 최적의 축척을 실시간으로 도출.

---

## 🕵️‍♂️ 딥 다이브 추론 로그 해석 가이드 (Log Interpretation)

시스템 우측 하단의 **'딥 다이브 추론 과정 디버거'**를 통해 엔진의 연산 과정을 투명하게 추적할 수 있습니다. 주요 로그의 의미는 다음과 같습니다.

### 1. 텍스트 분류 및 예외 객체 필터링
> `🧠 [LLM 분류 성공] 28개 단어 식별 (9.9초)`
> `⚠️ [LINE_46] 기하 추출 오류: unsupported DXF type: DIMENSION`
* **해석:** LLM이 도면 내 텍스트의 성격(방 이름, 가구 등)을 성공적으로 파악했습니다. `DIMENSION`(치수선)이나 `MLEADER`(지시선) 같은 비물리적 객체는 장벽 연산에서 의도적으로 무시(Pass)하여 노이즈를 방지합니다.

### 2. 기하학적 문(Door) 감지 및 보정
> `🚪 [LINE_118] 문 감지 (각도차: 90.0도) -> 직각(L자) 밀폐 장벽 생성`
* **해석:** 시맨틱 정보가 없더라도 기하학적으로 90도(1/4원)를 그리는 호(Arc)를 발견하여 **문**으로 자동 식별했습니다. 시각적으로는 원호를 유지하되, 내부 면적 계산 시에는 방의 모서리를 꽉 채우기 위해 **직각(L자) 장벽**으로 변환하여 밀폐한 과정을 나타냅니다.

### 3. 시맨틱 공간 매핑 (Space-to-Text Mapping)
> `✅ [방 매핑] Polygon #14 -> 'BEDROOM1 10'x11'-6"'`
> `✅ [방 매핑] Polygon #110 -> 'DININGAREA / KITCHEN 7'X8' / LIVINGAREA'`
* **해석:** 닫힌 다각형(Polygon) 내부에 위치한 텍스트 앵커를 매핑하여 방의 이름을 부여합니다. 벽체 없이 연결된 열린 공간(Open Space)의 경우, 여러 텍스트를 감지하여 하나의 거대한 공용 공간(`DINING / KITCHEN / LIVING`)으로 스마트하게 통합합니다.

### 4. 🌟 노이즈 격리 및 독립 보존 (Zero-Threshold 적용)
> `🧱 [독립 보존] 16 면적 -> UNKNOWN_SPACE 로 분리`
> `🧱 [독립 보존] 2653 면적 -> UNKNOWN_SPACE 로 분리`
* **해석:** 본 엔진의 핵심 철학인 **"불확실한 공간의 무분별한 합병 금지"**가 적용된 결과입니다. 잔해 흡수 한계를 0으로 설정하여, 면적이 16인 미세한 먼지 공간이나 2653인 벽체 내부 빈 공간(Wall Cavities)을 실제 방(BEDROOM 등)에 편입시키지 않습니다. 
* 이를 통해 오직 유의미한 공간만이 최종 지식 그래프(JSON)의 노드로 추출됩니다.

---

## 🚀 How to Run (실행 방법)

1. Python 환경(3.8 이상 권장)에서 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
2. 프로젝트 루트 경로에 .env 파일을 생성하고 OpenAI API 키를 입력합니다.
   ```bash
   OPENAI_API_KEY="sk-..."
   ```
3. Streamlit 앱을 실행합니다.
   ```bash
   streamlit run app.py
   ```