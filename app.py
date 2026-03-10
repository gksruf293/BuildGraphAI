"""
AI BuildGraph Precision Pro - Grow & Prune Engine
------------------------------------------------------
고객 제안 알고리즘 적용: 
가구를 하나씩 승격(Grow)시키며 벽을 연장하고, 
스스로 Dead End를 포함하는 가벽은 하나씩 강등(Prune)시키는 무결점 알고리즘입니다.
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
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import unary_union, polygonize
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. 초기화 및 환경 설정
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI BuildGraph Ultimate Engine", layout="wide")

# ==========================================
# 2. RAG 프롬프트 및 수석 건축가 LLM 추론 엔진
# ==========================================
def generate_architectural_knowledge_graph(api_key, raw_graph_data):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    
    context_str = "도면 공간 기초 데이터:\n"
    for node in raw_graph_data:
        context_str += f"- [ID: {node['id']}] 이름: {node['name']}, 면적: {node.get('area_sqft', 0)} sqft\n"
        context_str += f"  > 🚶 물리적 출입 가능 방: {node.get('accessible_to', [])}\n"
        context_str += f"  > 🧱 단순 맞닿은 방 (벽으로 막힘): {node.get('touching_to', [])}\n"

    system_prompt = """
    당신은 수석 건축 설계사입니다. 공간 데이터를 분석하여 건축 설계 지식 그래프를 생성하세요.
    반드시 아래 JSON 스키마를 엄격히 준수하여 응답하세요.
    {
      "nodes": [ {"id": "ROOM_X", "name": "...", "area_sqft": 100, "privacy_level": "PRIVATE", "primary_function": "RESIDENTIAL", "ventilation_req": "HIGH"} ],
      "edges": [ {"source": "ROOM_X", "target": "ROOM_Y", "type": "FUNCTIONAL", "relation": "REQUIRES_ACCESS_TO", "reason": "이유 설명"} ]
    }
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o", response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"다음 데이터를 분석하여 건축 설계 지식 그래프를 만들어주세요.\n{context_str}"}],
            temperature=0.1
        )
        return json.loads(res.choices[0].message.content.strip())
    except: return None

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def parse_imperial_dimensions(text):
    if not text or not isinstance(text, str): return None
    try:
        matches = re.findall(r"(\d+)'(?:-?(\d+)\"?)?", text.replace(' ', '').upper())
        dims = [int(m[0]) * 12 + (int(m[1]) if m[1] else 0) for m in matches]
        return sorted(dims[:2]) if len(dims) >= 2 else None
    except: return None

def format_imperial(inches):
    if inches <= 0: return "0'"
    feet, rem = int(inches // 12), int(round(inches % 12))
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
# 4. 텍스트 분류기
# ==========================================
def classify_texts_with_llm(api_key, text_list, debug_logs):
    unique_texts = list(set([t['text'] for t in text_list]))
    class_map = {}
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            prompt = f"분류: ROOM, FIXTURE, DIMENSION. Texts: {json.dumps(unique_texts)}"
            res = client.chat.completions.create(
                model="gpt-4o-mini", response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "Return JSON with a key 'classifications'."}, {"role": "user", "content": prompt}], temperature=0
            )
            raw = json.loads(res.choices[0].message.content.strip())
            class_map = {item.get('text', ''): item.get('category', 'UNKNOWN') for item in raw.get('classifications', [])}
        except: pass

    if not class_map:
        for text in unique_texts:
            t_up = text.upper()
            if any(kw in t_up for kw in ['BED', 'LIVING', 'KITCHEN', 'TOILET', 'SHOP', 'DIN', 'ROOM', 'UP']): class_map[text] = 'ROOM'
            elif any(kw in t_up for kw in ['WARD', 'CB', 'RF', 'TV', 'SINK', 'CAB', 'BATH', 'SHOWER']): class_map[text] = 'FIXTURE'
            else: class_map[text] = 'DIMENSION'
    return class_map, {t['layer'] for t in text_list if class_map.get(t['text']) == 'FIXTURE'}

# ==========================================
# 5. 기하학 처리
# ==========================================
def process_entity_advanced(entity, eff_layer, fixture_layers, fixture_polygons, obj_id):
    try:
        o_type = 'WALL'
        if eff_layer in fixture_layers or any(w in eff_layer.upper() for w in ['FUR', 'FIX', 'WARD', 'CB', 'SINK']): o_type = 'FIXTURE'
        if any(w in eff_layer.upper() for w in ['DOOR', 'WIND']): o_type = 'DOOR'
        
        p = path.make_path(entity)
        vs = list(p.flattening(distance=2.0))
        if len(vs) < 2: return None, None, o_type
        geom = LineString([(v.x, v.y) for v in vs])

        if o_type == 'WALL':
            for f_poly in fixture_polygons:
                if f_poly.contains(geom): o_type = 'FIXTURE'; break

        if o_type == 'DOOR' and entity.dxftype() == 'ARC':
            center, radius = entity.dxf.center, entity.dxf.radius
            s, e = entity.dxf.start_angle, entity.dxf.end_angle
            p1 = (center.x + radius * math.cos(math.radians(s)), center.y + radius * math.sin(math.radians(s)))
            p2 = (center.x + radius * math.cos(math.radians(e)), center.y + radius * math.sin(math.radians(e)))
            return geom, LineString([p1, (center.x, center.y), p2]), o_type

        return geom, geom, o_type
    except: return None, None, 'WALL'

# ==========================================
# 6. 코어 엔진 (🌟 Grow & Prune: 하나씩 승격/강등 알고리즘)
# ==========================================
def get_true_dead_ends(all_lines):
    """네트워크에서 다른 선분과 맞닿지 않은 순수한 '끊긴 점(Dead End)'을 찾습니다."""
    dead_ends = []
    for i, line in enumerate(all_lines):
        b = line.boundary
        pts = [b] if b.geom_type == 'Point' else (list(b.geoms) if b.geom_type == 'MultiPoint' else [])
        
        for pt in pts:
            touches_other = False
            for j, other_line in enumerate(all_lines):
                if i == j: continue
                if other_line.distance(pt) < 1.0: # 공차 허용
                    touches_other = True
                    break
            if not touches_other:
                dead_ends.append(pt)
    return dead_ends

def calculate_rooms(_objects, _text_data, class_map, debug_logs, debris_threshold, progress_bar=None):
    debug_logs.append("\n🧱 [공간 구획 엔진 시작: Grow & Prune (하나씩 승격 및 가지치기)]")
    
    wall_lines = [obj['core_line'] for obj in _objects.values() if obj['type'] == 'WALL']
    door_lines = [obj['core_line'] for obj in _objects.values() if obj['type'] == 'DOOR']
    base_lines = wall_lines + door_lines
    all_fixtures = {obj['id']: obj['core_line'] for obj in _objects.values() if obj['type'] == 'FIXTURE'}
    
    # ---------------------------------------------------------
    # 💡 [PHASE 1: GROW] 불완전한 벽의 끝점에 닿는 가구를 차례대로 하나씩 승격
    # ---------------------------------------------------------
    debug_logs.append("\n🌱 [PHASE 1: Grow] 하나씩 승격: 벽의 끊긴 점에 닿는 가구를 이어붙입니다.")
    promoted_fids = set()
    growing = True
    
    while growing:
        growing = False
        current_network = base_lines + [all_fixtures[fid] for fid in promoted_fids]
        dead_ends = get_true_dead_ends(current_network)
        
        for fid, f_geom in all_fixtures.items():
            if fid in promoted_fids: continue
            
            touches_de = False
            for de in dead_ends:
                if f_geom.distance(de) < 1.0:
                    touches_de = True
                    break
            
            if touches_de:
                promoted_fids.add(fid)
                growing = True
                debug_logs.append(f"  ➕ [{fid}] 가벽 임시 승격 (벽의 끝점과 연결됨)")
                break # 핵심: 하나를 찾았으면 즉시 루프를 멈추고 Dead End를 다시 계산합니다 (One by one)
                
    debug_logs.append(f"  👉 총 {len(promoted_fids)}개의 가구가 임시 가벽으로 승격되었습니다.")
    
    # ---------------------------------------------------------
    # 💡 [PHASE 2: PRUNE] 승격된 가벽 중 스스로 Dead End를 포함하면 하나씩 강등
    # ---------------------------------------------------------
    debug_logs.append("\n✂️ [PHASE 2: Prune] 반복 가지치기: 스스로 튀어나온 끝점(Dead End)을 포함하는 가벽을 강등합니다.")
    pruning = True
    
    while pruning:
        pruning = False
        current_network = base_lines + [all_fixtures[fid] for fid in promoted_fids]
        dead_ends = get_true_dead_ends(current_network)
        
        for fid in list(promoted_fids):
            f_geom = all_fixtures[fid]
            has_de = False
            
            # 이 가구의 끝점(Boundary) 추출
            f_eps = f_geom.boundary
            f_pts = [f_eps] if f_eps.geom_type == 'Point' else (list(f_eps.geoms) if f_eps.geom_type == 'MultiPoint' else [])
            
            # 내 끝점이 전체 네트워크의 Dead End에 해당하는지 확인
            for f_pt in f_pts:
                for de in dead_ends:
                    if f_pt.distance(de) < 0.1:
                        has_de = True
                        break
                if has_de: break
            
            if has_de:
                promoted_fids.remove(fid)
                pruning = True
                debug_logs.append(f"  ➖ [{fid}] 가벽 강등 (연결 실패 및 스스로 Dead End 노출)")
                break # 핵심: 하나를 강등했으면 즉시 루프를 멈추고 Dead End를 다시 계산합니다 (One by one)
                
    debug_logs.append(f"  ✨ 검증 완료! 최종적으로 {len(promoted_fids)}개의 가벽(WALL)이 완벽한 구조체로 확정되었습니다.")

    # ---------------------------------------------------------
    # 💡 [STEP 3] 최종 승격된 가벽을 시스템 객체로 등록
    # ---------------------------------------------------------
    inferred_walls = []
    inferred_walls_count = 0
    for fid in promoted_fids:
        new_id = f"INFERRED_WALL_{inferred_walls_count}"
        _objects[new_id] = {
            'geom': _objects[fid]['geom'], 'core_line': all_fixtures[fid], 
            'layer': 'AUTO_INFERRED_WALL', 'type': 'WALL', 'id': new_id
        }
        inferred_walls.append(all_fixtures[fid])
        inferred_walls_count += 1

    # ---------------------------------------------------------
    # 💡 [STEP 4] 가벽만 포함하여 순수 방바닥 구획 (일반 가구는 방바닥을 뚫지 않음)
    # ---------------------------------------------------------
    debug_logs.append(f"\n🧼 [STEP 4] 확정된 가벽만 포함하여 순수 바닥 구획 중...")
    
    final_wall_lines = wall_lines + inferred_walls
    room_barriers = final_wall_lines + door_lines 
    if not room_barriers: return []
    
    topo = unary_union(room_barriers).buffer(2.5, quad_segs=1, cap_style=3)
    raw_spaces = [Polygon(interior) for p in (list(topo.geoms) if hasattr(topo, 'geoms') else [topo]) for interior in p.interiors]
    
    physical_spaces = [s.buffer(2.5, join_style=2).simplify(0.5) for s in raw_spaces if s.buffer(2.5, join_style=2).simplify(0.5).area > 10]

    room_anchors = [{'point': Point(t['pos']), 'name': t['text']} for t in _text_data if class_map.get(t['text']) == 'ROOM']
    
    final_rooms, unnamed = [], []
    for space in physical_spaces:
        r_anchors = [a for a in room_anchors if space.contains(a['point'])]
        if r_anchors:
            names = sorted(list(set([a['name'] for a in r_anchors])))
            dims = [t['text'] for t in _text_data if space.contains(Point(t['pos'])) and class_map.get(t['text']) == 'DIMENSION']
            final_rooms.append({'id': f"ROOM_{len(final_rooms)}", 'geom': space, 'name': " / ".join(names), 'explicit_dim': " ".join(dims), 'type': 'NAMED'})
        else: unnamed.append(space)

    # 자투리(Debris) 흡수
    unnamed_formatted = []
    for j, shard in enumerate(unnamed):
        if shard.area < debris_threshold:
            best_r, max_overlap = None, -1
            buffered_s = shard.buffer(0.5)
            for r in final_rooms:
                overlap = buffered_s.intersection(r['geom']).area
                if overlap > max_overlap: max_overlap, best_r = overlap, r
            if best_r and max_overlap > 0:
                best_r['geom'] = unary_union([best_r['geom'], shard]).simplify(0.1)
                continue
        unnamed_formatted.append({'id': f"UNKNOWN_{j}", 'geom': shard, 'name': 'UNKNOWN_SPACE', 'type': 'UNKNOWN'})

    results = final_rooms + unnamed_formatted

    # ---------------------------------------------------------
    # [STEP 5] 시각화용 가구 폴리곤 오버레이 생성
    # ---------------------------------------------------------
    fixture_lines_original = [obj['core_line'] for obj in _objects.values() if obj['type'] == 'FIXTURE']
    fixture_spaces = []
    fixture_anchors = [{'point': Point(t['pos']), 'name': t['text']} for t in _text_data if class_map.get(t['text']) == 'FIXTURE']
    fixture_union = unary_union(fixture_lines_original).buffer(2.0) if fixture_lines_original else Polygon()
    
    if fixture_lines_original:
        fixture_polys = list(polygonize(fixture_union))
        for i, f_poly in enumerate(fixture_polys):
            if f_poly.area < 5: continue
            f_anchors = [a for a in fixture_anchors if f_poly.contains(a['point'])]
            name = " / ".join(sorted(list(set([a['name'] for a in f_anchors])))) if f_anchors else "Built-in Fixture"
            fixture_spaces.append({'id': f"FIXTURE_{i}", 'geom': f_poly, 'name': name, 'type': 'FIXTURE_SPACE'})

    results.extend(fixture_spaces)
    
    # ---------------------------------------------------------
    # [STEP 6] 방 인접성 및 출입 관계 추론
    # ---------------------------------------------------------
    wall_union = unary_union(final_wall_lines) if final_wall_lines else Polygon()
    for r in results: r['accessible_to'] = []; r['touching_to'] = []

    named_rooms = [r for r in results if r['type'] == 'NAMED']
    for i, r1 in enumerate(named_rooms):
        buf_r1_door = r1['geom'].buffer(15.0)
        buf_r1_wall = r1['geom'].buffer(10.0)
        for r2 in named_rooms: 
            if r1['id'] == r2['id']: continue
            has_door = any(d.intersects(buf_r1_door) and d.intersects(r2['geom'].buffer(15.0)) for d in door_lines)
            if has_door:
                r1['accessible_to'].append(r2['id'])
                if r1['id'] not in r2['accessible_to']: r2['accessible_to'].append(r1['id'])
                continue
            
            buf_r2_wall = r2['geom'].buffer(10.0)
            if buf_r1_wall.intersects(buf_r2_wall):
                open_path = buf_r1_wall.intersection(buf_r2_wall).difference(wall_union.buffer(5.0))
                if not open_path.is_empty and open_path.area > 15.0:
                    r1['accessible_to'].append(r2['id'])
                    if r1['id'] not in r2['accessible_to']: r2['accessible_to'].append(r1['id'])
                else:
                    if r1['geom'].buffer(3.0).intersects(r2['geom'].buffer(3.0)):
                        r1['touching_to'].append(r2['id'])
                        if r1['id'] not in r2['touching_to']: r2['touching_to'].append(r1['id'])

    return results

# ==========================================
# 7. 메인 UI (Streamlit 프론트엔드)
# ==========================================
st.sidebar.title("🛠️ AI BuildGraph All-in-One")
uploaded_file = st.sidebar.file_uploader("1️⃣ DXF 도면 업로드", type=['dxf'])

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 2️⃣ AI 수석 건축가 상담 (RAG)")
if st.sidebar.button("🚀 AI 도면 설계 분석 실행", type="primary"):
    if 'rooms' in st.session_state and st.session_state.rooms:
        with st.spinner("분석 중..."):
            rooms_for_llm = [{"id": r['id'], "name": r['name'], "area_sqft": round(r['geom'].area/144, 2), "accessible_to": [a for a in r.get('accessible_to',[]) if "ROOM" in a], "touching_to": [t for t in r.get('touching_to',[]) if "ROOM" in t]} for r in st.session_state.rooms if r['type'] == 'NAMED']
            rag_data = generate_architectural_knowledge_graph(OPENAI_API_KEY, rooms_for_llm)
            if rag_data: st.session_state['rag_data'] = rag_data; st.sidebar.success("✅ 설계 분석 완료!")
    else: st.sidebar.warning("도면을 업로드해주세요.")

show_rag = st.sidebar.checkbox("AI 설계 인사이트 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ 시각화 제어")
show_rooms = st.sidebar.checkbox("방 영역 표시", value=True)
show_unknown = st.sidebar.checkbox("이름 없는 공간/자투리", value=False)
show_walls = st.sidebar.checkbox("벽/문 표시", value=True)
show_fixtures = st.sidebar.checkbox("가구 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 물리 엔진 미세 조정")
debris_threshold = st.sidebar.slider("잔해 흡수 한계 면적", 0, 5000, 0, 50)
debug_inferred_walls = st.sidebar.checkbox("🚨 복원된 가벽 빨간색 강조", value=False)

if uploaded_file:
    cache_key = f"{uploaded_file.name}_{debris_threshold}"
    if 'current_cache_key' not in st.session_state or st.session_state.current_cache_key != cache_key:
        with st.status("🛠️ 파이프라인 가동 중...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            debug_logs = ["\n🚀 [엔진 가동] 파이프라인 시작"]
            doc = ezdxf.readfile(tmp_path)
            msp = doc.modelspace()
            os.remove(tmp_path)
            
            txts = [{'text': re.sub(r'\\[a-zA-Z0-9,-]+[^;]*;|[{} ]', '', ent.plain_text() if hasattr(ent, 'plain_text') else ent.dxf.text).strip(), 'pos': (ent.dxf.insert.x, ent.dxf.insert.y), 'layer': ent.dxf.layer} for ent in msp if ent.dxftype() in ('TEXT', 'MTEXT') and re.search(r'[a-zA-Z0-9]', re.sub(r'\\[a-zA-Z0-9,-]+[^;]*;|[{} ]', '', ent.plain_text() if hasattr(ent, 'plain_text') else ent.dxf.text).strip())]
            
            class_map, fixture_layers = classify_texts_with_llm(OPENAI_API_KEY, txts, debug_logs)
            
            all_raw_lines = []
            for ent in msp:
                if ent.dxftype() not in ('TEXT', 'MTEXT'):
                    for sub in (ent.virtual_entities() if ent.dxftype() == 'INSERT' else [ent]):
                        try:
                            vp = list(path.make_path(sub).flattening(distance=2.0))
                            if len(vp) >= 2: all_raw_lines.append(LineString([(v.x, v.y) for v in vp]))
                        except: pass
            
            all_raw_polys = sorted(list(polygonize(unary_union(all_raw_lines))), key=lambda x: x.area)
            fixture_polygons = [poly.buffer(2.0) for t_dict in txts if class_map.get(t_dict['text']) == 'FIXTURE' for poly in all_raw_polys if poly.contains(Point(t_dict['pos']))]

            objs = {}
            idx = 0
            for ent in msp:
                if ent.dxftype() not in ('TEXT', 'MTEXT'):
                    for sub in (ent.virtual_entities() if ent.dxftype() == 'INSERT' else [ent]):
                        eff_layer = ent.dxf.layer if sub.dxf.layer == '0' else sub.dxf.layer
                        geom_visual, core_line, o_type = process_entity_advanced(sub, eff_layer, fixture_layers, fixture_polygons, f"LINE_{idx}")
                        if geom_visual: objs[f"LINE_{idx}"] = {'geom': geom_visual, 'core_line': core_line, 'layer': eff_layer, 'type': o_type, 'id': f"LINE_{idx}"}
                        idx += 1
            
            rooms = calculate_rooms(objs, txts, class_map, debug_logs, debris_threshold)
            status.update(label="🎉 파이프라인 완료!", state="complete", expanded=False)
            st.session_state.update({'objects': objs, 'texts': txts, 'class_map': class_map, 'rooms': rooms, 'debug_logs': debug_logs, 'current_cache_key': cache_key})

    # --- 시각화 섹션 ---
    if 'rooms' in st.session_state and st.session_state.rooms:
        final_scale = 1.0 # 스케일 생략
        fig = go.Figure()
        palette = pcolors.qualitative.Pastel

        for i, r in enumerate(st.session_state.rooms):
            if r['type'] == 'FIXTURE_SPACE': continue
            if r['type'] == 'UNKNOWN' and not show_unknown: continue
            if r['type'] == 'NAMED' and not show_rooms: continue
            
            fill_color = palette[i % len(palette)] if r['type'] == 'NAMED' else 'rgba(150, 150, 150, 0.3)'
            for poly in (list(r['geom'].geoms) if r['geom'].geom_type in ('MultiPolygon', 'GeometryCollection') else [r['geom']]):
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.coords.xy
                    fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='toself', fillcolor=fill_color, line=dict(width=0), name=r['name'], opacity=0.8))
            if r['type'] == 'NAMED': fig.add_annotation(x=r['geom'].centroid.x, y=r['geom'].centroid.y, text=f"<b>{r['name']}</b>", showarrow=False, font=dict(size=10, color="black"))

        for i, r in enumerate(st.session_state.rooms):
            if r['type'] != 'FIXTURE_SPACE' or not show_fixtures: continue
            for poly in (list(r['geom'].geoms) if r['geom'].geom_type in ('MultiPolygon', 'GeometryCollection') else [r['geom']]):
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.coords.xy
                    fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='toself', fillcolor='rgba(243, 156, 18, 0.7)', line=dict(width=1, color='#d35400'), name=r['name'], opacity=1.0))

        for obj in st.session_state.objects.values():
            otype = obj['type']
            if (otype in ('WALL', 'DOOR') and not show_walls) or (otype == 'FIXTURE' and not show_fixtures): continue
            
            color = '#2c3e50'
            width = 2
            if otype == 'DOOR': color = '#2980b9'
            elif otype == 'FIXTURE': color = '#e67e22'
            elif otype == 'WALL' and debug_inferred_walls and obj.get('layer') == 'AUTO_INFERRED_WALL':
                color = '#ff0000'; width = 4
                    
            x, y = obj['geom'].exterior.coords.xy if hasattr(obj['geom'], 'exterior') else obj['geom'].xy
            fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='none', line=dict(color=color, width=width), showlegend=False))

        rag_data = st.session_state.get('rag_data')
        if rag_data and show_rag and 'edges' in rag_data:
            centroids = {r['id']: (r['geom'].centroid.x, r['geom'].centroid.y) for r in st.session_state.rooms}
            for edge in rag_data.get('edges', []):
                if edge.get('source') in centroids and edge.get('target') in centroids:
                    x0, y0 = centroids[edge.get('source')]; x1, y1 = centroids[edge.get('target')]
                    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowcolor='red')

        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1, visible=False), xaxis=dict(visible=False), plot_bgcolor='white', height=850)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🕵️‍♂️ 딥 다이브 추론 과정 디버거 (로그를 확인하세요!)", expanded=True):
            st.code("\n".join(st.session_state.debug_logs))