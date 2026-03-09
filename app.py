import streamlit as st
import ezdxf
import re
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

# 1. 초기화 및 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI BuildGraph Precision Pro v9", layout="wide")

# ==========================================
# 2. 유틸리티 및 치수 계산
# ==========================================
def parse_imperial_dimensions(text):
    if text is None or not isinstance(text, str) or text.strip() == "": return None
    try:
        matches = re.findall(r"(\d+)'(?:-?(\d+)\"?)?", text.replace(' ', '').upper())
        dims = [int(m[0]) * 12 + (int(m[1]) if m[1] else 0) for m in matches]
        return sorted(dims[:2]) if len(dims) >= 2 else None
    except: return None

def format_imperial(inches):
    if inches <= 0: return "0'"
    feet, rem = int(inches // 12), int(round(inches % 12))
    if rem == 12: feet += 1; rem = 0
    return f"{feet}'-{rem}\"" if rem else f"{feet}'"

def calculate_cad_dimensions(geom):
    if geom.is_empty: return [0, 0]
    rect = geom.minimum_rotated_rectangle
    if rect.geom_type != 'Polygon': return [0, 0]
    coords = list(rect.exterior.coords)
    w = Point(coords[0]).distance(Point(coords[1]))
    h = Point(coords[1]).distance(Point(coords[2]))
    return sorted([w, h])

# ==========================================
# 3. 지능형 LLM 분류 (가구 레이어 식별)
# ==========================================
def classify_texts_with_llm(api_key, text_list):
    if not api_key or not text_list: return {}, set()
    client = OpenAI(api_key=api_key)
    unique_texts = list(set([t['text'] for t in text_list]))
    
    prompt = f"""
    건축 도면 텍스트 분류기입니다. 다음을 'ROOM', 'FIXTURE', 'DIMENSION' 중 하나로 분류하여 JSON으로 응답.
    [Texts]: {json.dumps(unique_texts)}
    규칙:
    - ROOM: 사람이 활동하는 방 (BEDROOM1 10'x11' 처럼 치수가 섞여도 ROOM으로 분류)
    - FIXTURE: 가구/설비 (WARDROBE, CB, RF, TV, BED, SINK, CAB 등)
    - DIMENSION: 치수 정보만 있는 순수 숫자
    응답 형식: {{"classifications": [{{"text": "BEDROOM1", "category": "ROOM"}}]}}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini", response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
            temperature=0
        )
        content = re.sub(r'```json|```', '', res.choices[0].message.content.strip())
        raw_res = json.loads(content)
        class_map = {item.get('text', ''): item.get('category', 'UNKNOWN') for item in raw_res.get('classifications', []) if isinstance(item, dict)}
        
        # 🌟 가구 텍스트가 존재하는 레이어를 통째로 가구로 지정하기 위해 수집
        fixture_layers = {t['layer'] for t in text_list if class_map.get(t['text']) == 'FIXTURE'}
        return class_map, fixture_layers
    except: return {}, set()

# ==========================================
# 4. 🌟 기하학 처리 (가구 내 문 오인 방지 로직)
# ==========================================
def get_layer_type_v9(layer, fixture_layers):
    if layer in fixture_layers: return 'FIXTURE'
    layer_up = layer.upper()
    if any(w in layer_up for w in ['DIM', 'ANNO', 'MARK', 'TEXT', 'TXT']): return 'DIM_LINE'
    if any(w in layer_up for w in ['FUR', 'FIX', 'WARD', 'CB', 'BED', 'SINK', 'CAB', 'APP']): return 'FIXTURE'
    if any(w in layer_up for w in ['DOOR', 'WIND']): return 'DOOR'
    return 'WALL'

def process_entity_v9(entity, eff_layer, fixture_layers):
    try:
        o_type = get_layer_type_v9(eff_layer, fixture_layers)
        
        # 🌟 버그 수정: 객체가 이미 '가구(FIXTURE)'로 판명되었다면, 1/4원이더라도 문으로 변환하지 않고 그대로 둠!
        if o_type == 'FIXTURE':
            p = path.make_path(entity)
            vs = list(p.flattening(distance=2.0))
            if len(vs) < 2: return None, o_type
            return LineString([(v.x, v.y) for v in vs]), o_type

        # 🌟 1/4원 문 강제 변환: 가구가 아닌 경우에만 약 90도의 ARC를 DOOR로 취급
        if entity.dxftype() == 'ARC':
            angle_diff = abs(entity.dxf.end_angle - entity.dxf.start_angle) % 360
            if 80 <= angle_diff <= 100:
                o_type = 'DOOR'
                p = path.make_path(entity)
                vs = list(p.flattening(distance=5.0))
                if len(vs) >= 2:
                    # 곡선을 배제하고 양 끝을 잇는 직선(현)으로 방 밀폐
                    return LineString([(vs[0].x, vs[0].y), (vs[-1].x, vs[-1].y)]), o_type

        p = path.make_path(entity)
        vs = list(p.flattening(distance=2.0))
        if len(vs) < 2: return None, o_type
        return LineString([(v.x, v.y) for v in vs]), o_type
    except: return None, 'WALL'

# ==========================================
# 5. 핵심 엔진: 공간 분할 및 그래프 생성
# ==========================================
def calculate_rooms_v9(_objects, _text_data, class_map):
    debug_logs = []
    
    # 장벽 생성 (가구 철저히 제외)
    barriers = [obj['core_line'] for obj in _objects.values() if obj['type'] in ('WALL', 'DOOR')]
    if not barriers: return [], debug_logs

    topology = unary_union(barriers).buffer(2.5, quad_segs=1, cap_style=3)
    physical_spaces = []
    polys = list(topology.geoms) if hasattr(topology, 'geoms') else [topology]
    for p in polys:
        for interior in p.interiors:
            poly = Polygon(interior).simplify(1.0)
            if poly.area > 500: physical_spaces.append(poly)

    room_anchors = [{'point': Point(t['pos']), 'name': t['text']} for t in _text_data if class_map.get(t['text']) == 'ROOM']
    final_rooms, unnamed = [], []
    
    for i, space in enumerate(physical_spaces):
        anchors = [a for a in room_anchors if space.contains(a['point'])]
        if not anchors:
            unnamed.append(space); continue
        
        names = sorted(list(set([a['name'] for a in anchors])))
        dims = [t['text'] for t in _text_data if space.contains(Point(t['pos'])) and class_map.get(t['text']) == 'DIMENSION']
        
        final_rooms.append({
            'id': f"ROOM_{len(final_rooms)}", 'geom': space, 
            'name': " / ".join(names), 'explicit_dim': " ".join(dims), 'type': 'NAMED'
        })

    # 잔해 흡수
    results = final_rooms.copy()
    for j, shard in enumerate(unnamed):
        if shard.area < 25000:
            best_neighbor, max_overlap = None, -1
            search_area = shard.buffer(10.0)
            for r in results:
                overlap = search_area.intersection(r['geom']).length
                weight = 2.0 if r['type'] == 'NAMED' else 1.0
                if (overlap * weight) > max_overlap: max_overlap = overlap * weight; best_neighbor = r
            if best_neighbor and max_overlap > 0:
                best_neighbor['geom'] = unary_union([best_neighbor['geom'], shard])
                continue
        results.append({'id': f"ROOM_{len(results)}", 'geom': shard, 'name': 'Unknown/Corridor', 'type': 'UNKNOWN'})

    # 인접성(Edges) 계산
    for r1 in results:
        r1['adjacent_to'] = [r2['id'] for r2 in results if r1['id'] != r2['id'] and r1['geom'].buffer(5.0).intersects(r2['geom'])]

    return results, debug_logs

# ==========================================
# 6. 메인 UI (동적 스케일링 지원)
# ==========================================
st.sidebar.title("🛠️ AI BuildGraph Dynamic Pro")
uploaded_file = st.sidebar.file_uploader("DXF 업로드", type=['dxf'])

# 레이어 토글
st.sidebar.subheader("👁️ 시각화 레이어 토글")
show_rooms = st.sidebar.checkbox("방 영역 표시", value=True)
show_walls = st.sidebar.checkbox("벽체 및 문 표시", value=True)
show_fixtures = st.sidebar.checkbox("가구 및 설비 표시", value=True)

if uploaded_file:
    if 'parsed_file' not in st.session_state or st.session_state.parsed_file != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
            tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
        doc = ezdxf.readfile(tmp_path); msp = doc.modelspace(); os.remove(tmp_path)
        
        txts = []
        for ent in msp:
            if ent.dxftype() in ('TEXT', 'MTEXT'):
                t = ent.plain_text() if hasattr(ent, 'plain_text') else ent.dxf.text
                t = re.sub(r'\\[a-zA-Z0-9,-]+[^;]*;|[{} ]', '', t.replace('\\P', ' ')).strip()
                if t and re.search(r'[a-zA-Z0-9]', t):
                    txts.append({'text': t, 'pos': (ent.dxf.insert.x, ent.dxf.insert.y), 'layer': ent.dxf.layer})
        
        class_map, fixture_layers = classify_texts_with_llm(OPENAI_API_KEY, txts)
        
        objs = {}
        for ent in msp:
            if ent.dxftype() not in ('TEXT', 'MTEXT'):
                parent_layer = ent.dxf.layer
                sub_entities = ent.virtual_entities() if ent.dxftype() == 'INSERT' else [ent]
                for sub in sub_entities:
                    eff_layer = parent_layer if sub.dxf.layer == '0' else sub.dxf.layer
                    ln, o_type = process_entity_v9(sub, eff_layer, fixture_layers)
                    if ln:
                        objs[f"LINE_{len(objs)}"] = {'geom': ln.buffer(1.5, cap_style=3), 'core_line': ln, 'layer': eff_layer, 'type': o_type}
        
        st.session_state.update({
            'objects': objs, 'texts': txts, 'class_map': class_map, 
            'parsed_file': uploaded_file.name, 'trigger_recalc': True
        })

    # 무거운 공간 기하 연산은 한 번만 수행하여 캐싱
    if st.session_state.get('trigger_recalc'):
        rooms, logs = calculate_rooms_v9(st.session_state.objects, st.session_state.texts, st.session_state.class_map)
        st.session_state.update({'rooms': rooms, 'l_logs': logs, 'trigger_recalc': False})

    # ==========================================
    # 🌟 7. 동적 스케일 캘리브레이션 (UI 연동)
    # ==========================================
    if 'rooms' in st.session_state and st.session_state.rooms:
        calibratable_rooms = []
        for r in st.session_state.rooms:
            real_dim = parse_imperial_dimensions(r['name']) or parse_imperial_dimensions(r.get('explicit_dim', ''))
            if real_dim:
                cad_dim = calculate_cad_dimensions(r['geom'])
                if cad_dim[0] > 0 and (r['geom'].area / r['geom'].minimum_rotated_rectangle.area > 0.7):
                    ratio = (real_dim[0]/cad_dim[0] + real_dim[1]/cad_dim[1])/2
                    calibratable_rooms.append({"name": r['name'], "ratio": ratio, "explicit": real_dim, "cad": cad_dim})
        
        final_scale = 1.0
        if calibratable_rooms:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📐 동적 스케일 캘리브레이션")
            all_names = [cr['name'] for cr in calibratable_rooms]
            
            # 🌟 오픈 공간 자동 제외 (주방, 거실 등은 기본 체크 해제)
            unreliable_keywords = ['KITCHEN', 'LIVING', 'DINING', 'SHOP', 'LOBBY', 'ENTRANCE']
            default_sel = [n for n in all_names if not any(kw in n.upper() for kw in unreliable_keywords)]
            if not default_sel: default_sel = all_names # 모두 오픈 공간일 경우 대비
            
            # 사용자 선택에 따라 실시간으로 스케일이 바뀜
            selected_names = st.sidebar.multiselect(
                "스케일 기준 방 선택 (밀폐된 침실, 화장실 권장)", 
                options=all_names, 
                default=default_sel,
                help="이곳에서 방을 켜고 끄면 도면 전체의 추정 면적(Est)이 즉시 재계산됩니다."
            )
            
            active_ratios = [cr['ratio'] for cr in calibratable_rooms if cr['name'] in selected_names]
            if active_ratios:
                final_scale = np.median(active_ratios)
                
            # 디버깅 표 표시
            active_table = [{"Room": cr['name'], "Scale Factor": round(cr['ratio'], 4)} for cr in calibratable_rooms if cr['name'] in selected_names]
            if active_table:
                st.sidebar.table(active_table)
            st.sidebar.success(f"🎯 현재 적용된 축척: 1 CAD Unit = {final_scale:.4f} Inches")

        # ------------------------------------------
        # 8. 최종 렌더링
        # ------------------------------------------
        fig = go.Figure()
        palette = pcolors.qualitative.Pastel

        if show_rooms:
            for i, r in enumerate(st.session_state.rooms):
                geom = r['geom']
                cad_dims = calculate_cad_dimensions(geom)
                # 선택된 final_scale에 따라 동적으로 Est 값이 렌더링됨
                calc_dim = f"{format_imperial(cad_dims[0]*final_scale)} x {format_imperial(cad_dims[1]*final_scale)}"
                
                polys = list(geom.geoms) if geom.geom_type in ('MultiPolygon', 'GeometryCollection') else [geom]
                for poly in polys:
                    if poly.geom_type == 'Polygon':
                        x, y = poly.exterior.coords.xy
                        fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='toself', fillcolor=palette[i % len(palette)], line=dict(width=0), name=r['name'], hoverinfo='text', text=f"<b>{r['name']}</b><br>Est: {calc_dim}", opacity=0.6))
                fig.add_annotation(x=geom.centroid.x, y=geom.centroid.y, text=f"<b>{r['name']}</b>", showarrow=False, font=dict(size=10))

        # 물리 레이어 렌더링
        l_configs = {'WALL': '#2c3e50', 'DOOR': '#2980b9', 'FIXTURE': '#f39c12'}
        for otype, color in l_configs.items():
            if (otype in ('WALL', 'DOOR') and not show_walls) or (otype == 'FIXTURE' and not show_fixtures): continue
            x_all, y_all = [], []
            for obj in st.session_state.objects.values():
                if obj['type'] == otype:
                    x, y = obj['geom'].exterior.coords.xy
                    x_all.extend(list(x) + [None]); y_all.extend(list(y) + [None])
            if x_all: fig.add_trace(go.Scatter(x=x_all, y=y_all, fill='toself', fillcolor=color, line=dict(color=color, width=1), name=otype, hoverinfo='none', mode='lines'))

        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1, visible=False), xaxis=dict(visible=False), plot_bgcolor='white', height=800, margin=dict(l=0, r=0, t=0, b=0), dragmode='zoom')
        st.plotly_chart(fig, use_container_width=True)
        
        # 동적 스케일이 반영된 Graph JSON 추출
        graph_json = json.dumps([{"id": r['id'], "name": r['name'], "adjacent": r['adjacent_to'], "area_sqft": round(r['geom'].area * (final_scale**2) / 144, 2)} for r in st.session_state.rooms], indent=2)
        st.download_button("📥 Export Graph JSON (동적 면적 반영)", data=graph_json, file_name="spatial_graph.json")