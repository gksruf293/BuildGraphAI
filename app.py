"""
AI BuildGraph Precision Pro - Spatial Reasoning Engine
------------------------------------------------------
이 모듈은 2D CAD 도면(DXF)을 업로드받아, LLM(gpt-4o-mini) 기반의 시맨틱 텍스트 분석과
Shapely 기반의 정밀 기하학(Geometry) 연산을 통해 도면 내 공간(Room)을 자동으로 분할하고,
각 공간의 면적 및 인접성(Connectivity)을 포함한 지식 그래프(Knowledge Graph) JSON을 생성합니다.

주요 알고리즘:
1. LLM Semantic Classification (가구/방/치수 자동 분류 및 자체 Fallback 엔진)
2. Arc-to-L-Shape Conversion (문의 궤적을 직각 장벽으로 변환하여 면적 손실 방지)
3. Ray-casting & Polygon Union (벽체와 문을 결합하여 닫힌 다각형 공간 추출)
4. Dynamic Scale Calibration (사용자 선택 기반 실시간 도면 축척 보정)
"""

import streamlit as st
import ezdxf
import re
import math
import tempfile
import os
import time
import json
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from ezdxf import path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. 초기화 및 환경 설정
# ==========================================
# 환경 변수에서 OpenAI API 키를 안전하게 로드합니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI BuildGraph Ultimate Engine", layout="wide")

# ==========================================
# 2. 유틸리티 및 치수 계산 (Utility Functions)
# ==========================================
def parse_imperial_dimensions(text):
    """
    도면 내의 텍스트에서 인치 단위의 치수(예: 10'x11'-6")를 파싱하여 숫자로 반환합니다.
    Returns: [가로 인치, 세로 인치] (오름차순 정렬)
    """
    if text is None or not isinstance(text, str) or text.strip() == "": return None
    try:
        # 정규표현식을 사용하여 피트(')와 인치(") 정보를 추출
        matches = re.findall(r"(\d+)'(?:-?(\d+)\"?)?", text.replace(' ', '').upper())
        dims = [int(m[0]) * 12 + (int(m[1]) if m[1] else 0) for m in matches]
        return sorted(dims[:2]) if len(dims) >= 2 else None
    except: return None

def format_imperial(inches):
    """숫자(인치)를 다시 사람이 읽기 쉬운 피트-인치 문자열로 변환합니다."""
    if inches <= 0: return "0'"
    feet, rem = int(inches // 12), int(round(inches % 12))
    if rem == 12: feet += 1; rem = 0
    return f"{feet}'-{rem}\"" if rem else f"{feet}'"

def calculate_cad_dimensions(geom):
    """
    Shapely 다각형(Polygon)의 기하학적 최소 회전 직사각형(Minimum Rotated Rectangle)을 
    구하여 CAD Unit 기준의 [가로, 세로] 길이를 도출합니다.
    """
    if geom.is_empty: return [0, 0]
    rect = geom.minimum_rotated_rectangle
    if rect.geom_type != 'Polygon': return [0, 0]
    coords = list(rect.exterior.coords)
    # 직사각형의 두 인접한 변의 길이를 계산
    w = Point(coords[0]).distance(Point(coords[1]))
    h = Point(coords[1]).distance(Point(coords[2]))
    return sorted([w, h])

# ==========================================
# 3. 지능형 텍스트 분류기 (LLM + Fallback)
# ==========================================
def classify_texts_with_llm(api_key, text_list, debug_logs):
    """
    도면 내 모든 텍스트를 수집하여 LLM을 통해 [ROOM, FIXTURE, DIMENSION]으로 의미론적 분류를 수행합니다.
    API 응답 지연이나 JSON 파싱 에러 발생 시, 정규식 기반의 하드코딩 룰(Fallback)이 즉시 개입하여 시스템 중단을 막습니다.
    """
    unique_texts = list(set([t['text'] for t in text_list]))
    class_map = {}
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            prompt = f"""
            다음 텍스트를 'ROOM', 'FIXTURE', 'DIMENSION' 중 하나로 분류하여 JSON으로 응답.
            [Texts]: {json.dumps(unique_texts)}
            - ROOM: 사람이 활동하는 방 (치수가 섞여도 ROOM. 예: BEDROOM1 10'x11')
            - FIXTURE: 가구/설비 (WARDROBE, CB, TV, BED, SINK 등)
            - DIMENSION: 순수 숫자
            응답 포맷: {{"classifications": [{{"text": "단어", "category": "ROOM"}}]}}
            """
            start_t = time.time()
            res = client.chat.completions.create(
                model="gpt-4o-mini", response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
                temperature=0
            )
            # Markdown 찌꺼기(```json) 제거 후 파싱
            raw = json.loads(re.sub(r'```json|```', '', res.choices[0].message.content.strip()))
            class_map = {item.get('text', ''): item.get('category', 'UNKNOWN') for item in raw.get('classifications', []) if isinstance(item, dict)}
            debug_logs.append(f"🧠 [LLM 분류 성공] {len(class_map)}개 단어 식별 ({time.time()-start_t:.1f}초)")
        except Exception as e:
            debug_logs.append(f"⚠️ [LLM 에러] {str(e)}")

    # 🛡️ Fallback Logic: LLM 분류가 실패했거나 결과가 비어있을 경우 안전장치 가동
    if not class_map or len(class_map) == 0:
        for text in unique_texts:
            t_up = text.upper()
            if any(kw in t_up for kw in ['BED', 'LIVING', 'KITCHEN', 'TOILET', 'SHOP', 'BATH', 'DIN', 'ROOM', 'UP', 'ENTRANCE']):
                class_map[text] = 'ROOM'
            elif any(kw in t_up for kw in ['WARD', 'CB', 'RF', 'TV', 'SINK', 'CAB', 'APP', 'SOFA', 'CHAIR']):
                class_map[text] = 'FIXTURE'
            elif re.search(r'\d', text):
                class_map[text] = 'DIMENSION'
        debug_logs.append(f"🛡️ [폴백 엔진 가동] 하드코딩 룰로 텍스트 강제 식별 완료.")

    # 분류 결과를 바탕으로 '가구 전용 레이어'를 추출하여 반환 (기하학 연산 시 장벽에서 제외하기 위함)
    fixture_layers = {t['layer'] for t in text_list if class_map.get(t['text']) == 'FIXTURE'}
    return class_map, fixture_layers

# ==========================================
# 4. 기하학 처리기 (수학적 1/4원 판별 및 L자 변환)
# ==========================================
def get_layer_type(layer, fixture_layers, layer_overrides):
    """레이어 이름과 LLM의 분류 결과를 종합하여 선분의 논리적 타입(WALL, DOOR, FIXTURE)을 결정합니다."""
    if layer in layer_overrides and layer_overrides[layer] != "AUTO":
        return layer_overrides[layer]
        
    if layer in fixture_layers: return 'FIXTURE' # LLM이 가구로 판단한 레이어는 최우선 적용
    layer_up = layer.upper()
    if any(w in layer_up for w in ['DIM', 'ANNO', 'MARK', 'TEXT', 'TXT']): return 'DIM_LINE'
    if any(w in layer_up for w in ['FUR', 'FIX', 'WARD', 'CB', 'BED', 'SINK', 'CAB', 'APP']): return 'FIXTURE'
    if any(w in layer_up for w in ['DOOR', 'WIND']): return 'DOOR'
    return 'WALL'

def process_entity_advanced(entity, eff_layer, fixture_layers, debug_logs, obj_id, id_overrides, layer_overrides):
    """
    도면의 각 요소를 분석합니다. 특히 문(ARC)의 경우,
    시각적 렌더링용으로는 원본 곡선을 유지하고, 공간 분할 연산용으로는 직각(L자) 장벽을 생성하여 반환합니다.
    """
    try:
        o_type = get_layer_type(eff_layer, fixture_layers, layer_overrides)
        
        # 사용자 수동 오버라이드 (강제 문 지정 등)
        is_forced_door = False
        if obj_id in id_overrides:
            o_type = id_overrides[obj_id]
            if o_type == 'DOOR': is_forced_door = True

        # 가구(FIXTURE)는 방 구획에 영향을 주지 않으므로 있는 그대로 반환
        if o_type == 'FIXTURE':
            p = path.make_path(entity)
            vs = list(p.flattening(distance=2.0))
            if len(vs) < 2: return None, None, o_type
            geom = LineString([(v.x, v.y) for v in vs])
            return geom, geom, o_type

        # 🚪 핵심 수학 로직: 1/4원(문) 인식 및 직각 장벽 변환
        if entity.dxftype() == 'ARC':
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            # 음수 각도 및 방향 전환을 고려한 360도 모듈러 연산
            angle_diff = (end_angle - start_angle) % 360
            
            # 대략 70~110도 사이의 원호는 문의 스윙 궤적으로 간주
            if (70 <= angle_diff <= 110) or is_forced_door or o_type == 'DOOR':
                o_type = 'DOOR'
                p = path.make_path(entity)
                
                # 1. 시각용(Visual): 원래의 예쁜 1/4 곡선을 유지
                vs_visual = list(p.flattening(distance=0.5))
                geom_visual = LineString([(v.x, v.y) for v in vs_visual]) if len(vs_visual) >= 2 else None
                
                # 2. 논리용(Logical): 면적이 깎이는 것을 막기 위해 중심점(Hinge)을 거쳐가는 L자 장벽 생성
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_rad = math.radians(start_angle)
                end_rad = math.radians(end_angle)
                
                pt_start = (center.x + radius * math.cos(start_rad), center.y + radius * math.sin(start_rad))
                pt_end = (center.x + radius * math.cos(end_rad), center.y + radius * math.sin(end_rad))
                
                # 시작점 -> 중심점 -> 끝점을 잇는 직각 폴리라인 구축
                core_line = LineString([pt_start, (center.x, center.y), pt_end])
                
                if geom_visual and core_line:
                    debug_logs.append(f"   🚪 [{obj_id}] 문 감지 (각도차: {angle_diff:.1f}도) -> 직각(L자) 밀폐 장벽 생성")
                    return geom_visual, core_line, o_type

        # 일반 직선 객체 처리
        p = path.make_path(entity)
        vs = list(p.flattening(distance=2.0))
        if len(vs) < 2: return None, None, o_type
        geom = LineString([(v.x, v.y) for v in vs])
        return geom, geom, o_type
    except Exception as e:
        debug_logs.append(f"   ⚠️ [{obj_id}] 기하 추출 오류: {str(e)}")
        return None, None, 'WALL'

# ==========================================
# 5. 코어 엔진 (공간 구획 및 그래프 연결성 추출)
# ==========================================
def calculate_rooms(_objects, _text_data, class_map, debug_logs, debris_threshold):
    debug_logs.append("\n🧱 [공간 구획 엔진 시작]")
    
    # 1. 물리적 장벽 구성 (가구 제외)
    barriers = [obj['core_line'] for obj in _objects.values() if obj['type'] in ('WALL', 'DOOR')]
    if not barriers: return []

    # 장벽들을 융합하고 미세한 틈새를 메우기 위해 2.5 유닛만큼 팽창(Buffer) 연산
    topology = unary_union(barriers).buffer(2.5, quad_segs=1, cap_style=3)
    physical_spaces = []
    polys = list(topology.geoms) if hasattr(topology, 'geoms') else [topology]
    for p in polys:
        # 벽체 내부의 빈 구멍(Interiors)을 실제 방 후보로 추출
        for interior in p.interiors:
            poly = Polygon(interior).simplify(1.0)
            if poly.area > 10: # 아주 미세한 먼지 노이즈 제거
                physical_spaces.append(poly)

    # 2. 앵커(방 이름) 매핑
    room_anchors = [{'point': Point(t['pos']), 'name': t['text']} for t in _text_data if class_map.get(t['text']) == 'ROOM']
    final_rooms, unnamed = [], []
    
    for i, space in enumerate(physical_spaces):
        # 공간(Polygon) 안에 방 이름 텍스트의 좌표(Point)가 포함되어 있는지 검사
        anchors = [a for a in room_anchors if space.contains(a['point'])]
        if not anchors:
            unnamed.append(space); continue
        
        # 공용 공간 병합 (예: 거실과 주방이 벽 없이 연결된 경우 'LIVING / KITCHEN'으로 이름 통합)
        names = sorted(list(set([a['name'] for a in anchors])))
        dims = [t['text'] for t in _text_data if space.contains(Point(t['pos'])) and class_map.get(t['text']) == 'DIMENSION']
        final_rooms.append({'id': f"ROOM_{len(final_rooms)}", 'geom': space, 'name': " / ".join(names), 'explicit_dim': " ".join(dims), 'type': 'NAMED'})
        debug_logs.append(f"   ✅ [방 매핑] Polygon #{i} -> '{final_rooms[-1]['name']}'")

    results = final_rooms.copy()
    
    # 3. 잔해 흡수 (Debris Absorption) 및 UNKNOWN 격리
    for j, shard in enumerate(unnamed):
        # 사용자가 설정한 임계값(기본 0~100) 이하의 파편만 인접 방에 편입 시도
        if shard.area < debris_threshold:
            best_n, max_o = None, -1
            search_area = shard.buffer(10.0)
            for r in results:
                # 잔해는 UNKNOWN끼리 합치지 않고 이름이 있는 'NAMED' 방에만 합침
                if r['type'] != 'NAMED': continue 
                overlap = search_area.intersection(r['geom']).length
                if overlap > max_o: max_o = overlap; best_n = r
            
            # 기하학적 융합(Union) 처리
            if best_n and max_o > 0:
                best_n['geom'] = unary_union([best_n['geom'], shard])
                debug_logs.append(f"   🧹 [잔해 청소] {int(shard.area)} 면적 -> '{best_n['name']}' 흡수")
                continue
        
        # 흡수되지 않은 큰 이름 없는 공간(기둥 내부, 복도 등)은 UNKNOWN 속성으로 격리 보존
        results.append({'id': f"UNKNOWN_{j}", 'geom': shard, 'name': 'UNKNOWN_SPACE', 'type': 'UNKNOWN'})
        debug_logs.append(f"   🧱 [독립 보존] {int(shard.area)} 면적 -> UNKNOWN_SPACE 로 분리")

    # 4. 연결성 (Graph Connectivity / Edge) 추출
    # 각 방이 5.0 유닛 이내로 인접해 있으면 서로 이동 가능한 노드로 간주
    for r1 in results:
        r1['adjacent_to'] = [r2['id'] for r2 in results if r1['id'] != r2['id'] and r1['geom'].buffer(5.0).intersects(r2['geom'])]

    return results

# ==========================================
# 6. 메인 UI (Streamlit 프론트엔드)
# ==========================================
st.sidebar.title("🛠️ AI BuildGraph Interactive")

uploaded_file = st.sidebar.file_uploader("DXF 업로드", type=['dxf'])

st.sidebar.subheader("👁️ 시각화 제어")
show_rooms = st.sidebar.checkbox("방 영역 표시 (NAMED)", value=True)
show_unknown = st.sidebar.checkbox("이름 없는 공간/벽 내부 (UNKNOWN)", value=False, help="벽체 내부 빈 공간이나 라벨이 없는 자투리 구역을 표시합니다.")
show_walls = st.sidebar.checkbox("벽체 및 문 표시", value=True)
show_fixtures = st.sidebar.checkbox("가구 및 설비 표시", value=True)
show_obj_ids = st.sidebar.checkbox("객체 ID 표시 (디버깅용)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 물리 엔진 미세 조정")
# 🌟 사용자 피드백 반영: 기본값을 0으로 설정하여 무분별한 잔해 흡수 방지
debris_threshold = st.sidebar.slider("잔해 흡수 한계 면적", min_value=0, max_value=5000, value=0, step=50, help="이 수치 이하의 공간만 옆 방으로 합쳐집니다. 0일 때 벽체 구분이 가장 완벽합니다.")
force_door_ids_str = st.sidebar.text_input("강제 문(DOOR) 지정 객체 ID", placeholder="예: LINE_118, LINE_121")

if uploaded_file:
    force_door_ids = [s.strip() for s in force_door_ids_str.split(",") if s.strip()]
    id_overrides = {fid: 'DOOR' for fid in force_door_ids}
    
    # 세션 상태(Cache) 관리를 통한 불필요한 재연산 방지
    cache_key = f"{uploaded_file.name}_{debris_threshold}_{force_door_ids_str}"
    
    if 'current_cache_key' not in st.session_state or st.session_state.current_cache_key != cache_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
            tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
        doc = ezdxf.readfile(tmp_path); msp = doc.modelspace(); os.remove(tmp_path)
        
        debug_logs = ["\n🚀 [엔진 가동] 파이프라인 시작"]
        
        # 파이프라인 Step 1: 텍스트 추출 및 LLM 분류
        txts = []
        for ent in msp:
            if ent.dxftype() in ('TEXT', 'MTEXT'):
                t = ent.plain_text() if hasattr(ent, 'plain_text') else ent.dxf.text
                t = re.sub(r'\\[a-zA-Z0-9,-]+[^;]*;|[{} ]', '', t.replace('\\P', ' ')).strip()
                if t and re.search(r'[a-zA-Z0-9]', t):
                    txts.append({'text': t, 'pos': (ent.dxf.insert.x, ent.dxf.insert.y), 'layer': ent.dxf.layer})
        
        class_map, fixture_layers = classify_texts_with_llm(OPENAI_API_KEY, txts, debug_logs)
        
        # 파이프라인 Step 2: 기하 객체 추출 및 블록(INSERT) 분해
        objs = {}
        idx = 0
        for ent in msp:
            if ent.dxftype() not in ('TEXT', 'MTEXT'):
                parent_layer = ent.dxf.layer
                sub_entities = ent.virtual_entities() if ent.dxftype() == 'INSERT' else [ent]
                for sub in sub_entities:
                    eff_layer = parent_layer if sub.dxf.layer == '0' else sub.dxf.layer
                    obj_id = f"LINE_{idx}"
                    layer_overrides = {} 
                    
                    # 튜플 언패킹을 통해 시각용(geom)과 논리연산용(core_line) 분리 수집
                    geom_visual, core_line, o_type = process_entity_advanced(sub, eff_layer, fixture_layers, debug_logs, obj_id, id_overrides, layer_overrides)
                    
                    if geom_visual and core_line:
                        objs[obj_id] = {'geom': geom_visual, 'core_line': core_line, 'layer': eff_layer, 'type': o_type, 'id': obj_id}
                    idx += 1
        
        # 파이프라인 Step 3: 공간 분할 및 인접성 그래프 도출
        rooms = calculate_rooms(objs, txts, class_map, debug_logs, debris_threshold)
        
        st.session_state.update({
            'objects': objs, 'texts': txts, 'class_map': class_map, 
            'rooms': rooms, 'debug_logs': debug_logs, 'current_cache_key': cache_key
        })

    # ==========================================
    # 7. 동적 스케일 캘리브레이션 (UI 연동)
    # ==========================================
    # 무거운 기하학 연산 없이 사용자의 입력에 따라 스케일만 동적으로 재계산합니다.
    if 'rooms' in st.session_state and st.session_state.rooms:
        calibratable = []
        for r in st.session_state.rooms:
            if r['type'] == 'UNKNOWN': continue # 이름 없는 공간은 캘리브레이션에서 원천 배제
            
            real_dim = parse_imperial_dimensions(r['name']) or parse_imperial_dimensions(r.get('explicit_dim', ''))
            if real_dim:
                cad_dim = calculate_cad_dimensions(r['geom'])
                # 기하학적 형태가 직사각형에 가깝고(70% 일치율) 유효한 치수가 있을 때만 활용
                if cad_dim[0] > 0 and (r['geom'].area / r['geom'].minimum_rotated_rectangle.area > 0.7):
                    ratio = (real_dim[0]/cad_dim[0] + real_dim[1]/cad_dim[1])/2
                    calibratable.append({"name": r['name'], "ratio": ratio})
        
        final_scale = 1.0
        if calibratable:
            st.sidebar.markdown("---")
            all_names = [cr['name'] for cr in calibratable]
            # 스케일 측정에 불리한 오픈 공간(Kitchen, Living)은 기본 선택에서 제외
            default_sel = [n for n in all_names if not any(k in n.upper() for k in ['KITCHEN', 'LIVING', 'DINING'])]
            if not default_sel: default_sel = all_names
            
            selected = st.sidebar.multiselect("📐 스케일 기준 방", options=all_names, default=default_sel)
            active_ratios = [cr['ratio'] for cr in calibratable if cr['name'] in selected]
            if active_ratios: final_scale = np.median(active_ratios)
            st.sidebar.success(f"🎯 적용 축척: 1 Unit = {final_scale:.4f} Inches")

        # ==========================================
        # 8. 최종 렌더링 (Plotly)
        # ==========================================
        fig = go.Figure()
        palette = pcolors.qualitative.Pastel

        for i, r in enumerate(st.session_state.rooms):
            # 레이어 표시 조건 분기
            if r['type'] == 'UNKNOWN' and not show_unknown: continue
            if r['type'] == 'NAMED' and not show_rooms: continue

            geom = r['geom']
            cad_dims = calculate_cad_dimensions(geom)
            calc_dim = f"{format_imperial(cad_dims[0]*final_scale)} x {format_imperial(cad_dims[1]*final_scale)}"
            
            # UNKNOWN 공간은 눈에 띄지 않는 반투명 회색 적용
            fill_color = palette[i % len(palette)] if r['type'] == 'NAMED' else 'rgba(150, 150, 150, 0.3)'
            
            polys = list(geom.geoms) if geom.geom_type in ('MultiPolygon', 'GeometryCollection') else [geom]
            for poly in polys:
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.coords.xy
                    fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='toself', fillcolor=fill_color, line=dict(width=0), name=r['name'], hoverinfo='text', text=f"<b>{r['name']}</b><br>Est: {calc_dim}", opacity=0.6))
            
            if r['type'] == 'NAMED':
                fig.add_annotation(x=geom.centroid.x, y=geom.centroid.y, text=f"<b>{r['name']}</b>", showarrow=False, font=dict(size=10, color="black"))

        # 물리 객체 렌더링 (선분)
        l_configs = {'WALL': '#2c3e50', 'DOOR': '#2980b9', 'FIXTURE': '#f39c12'}
        for otype, color in l_configs.items():
            if (otype in ('WALL', 'DOOR') and not show_walls) or (otype == 'FIXTURE' and not show_fixtures): continue
            x_all, y_all = [], []
            id_x, id_y, id_texts = [], [], []
            
            for obj in st.session_state.objects.values():
                if obj['type'] == otype:
                    # 렌더링 시에는 L자 장벽(core_line) 대신 시각용 1/4원 곡선(geom)을 사용
                    x, y = obj['geom'].exterior.coords.xy if hasattr(obj['geom'], 'exterior') else obj['geom'].xy
                    x_all.extend(list(x) + [None]); y_all.extend(list(y) + [None])
                    
                    if show_obj_ids:
                        id_x.append(obj['geom'].centroid.x); id_y.append(obj['geom'].centroid.y); id_texts.append(obj['id'])
                        
            if x_all: 
                fig.add_trace(go.Scatter(x=x_all, y=y_all, fill='none', line=dict(color=color, width=2), name=otype, hoverinfo='none', mode='lines'))
            if show_obj_ids and id_texts:
                fig.add_trace(go.Scatter(x=id_x, y=id_y, mode='text', text=id_texts, textfont=dict(size=8, color="red"), showlegend=False))

        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1, visible=False), xaxis=dict(visible=False), plot_bgcolor='white', height=800, margin=dict(l=0, r=0, t=0, b=0), dragmode='zoom')
        st.plotly_chart(fig, use_container_width=True)
        
        # 🌟 지식 그래프 JSON 추출 (면적 변환: 제곱인치 -> 평방피트)
        graph_json = json.dumps([{"id": r['id'], "name": r['name'], "adjacent": r['adjacent_to'], "area_sqft": round(r['geom'].area * (final_scale**2) / 144, 2)} for r in st.session_state.rooms if r['type'] == 'NAMED'], indent=2)
        st.download_button("📥 Export Graph JSON (동적 면적 반영)", data=graph_json, file_name="spatial_graph.json")

        with st.expander("🕵️‍♂️ 딥 다이브 추론 과정 디버거", expanded=True):
            st.code("\n".join(st.session_state.debug_logs))