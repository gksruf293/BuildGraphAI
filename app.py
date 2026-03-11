"""
AI BuildGraph Precision Pro - Ultimate Topological RAG Engine
------------------------------------------------------
[핵심 업데이트]
1. Grow & Prune Isolation: 문(Door)의 끝점을 벽의 단절(Dead End)로 오인하지 않도록 물리 엔진 분리.
2. Distance-based Top-2 Door Assignment: 억지 교차가 아닌, 문과 가장 가까운 방(Top 2)을 거리순으로 추적하여 완벽하게 문 소유권을 할당.
"""

import streamlit as st
import ezdxf
import re
import math
import tempfile
import os
import json
import plotly.graph_objects as go
import plotly.colors as pcolors
from ezdxf import path
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, polygonize
from dotenv import load_dotenv
from openai import OpenAI
from shapely.strtree import STRtree # 상단에 추가
import time
class RealTimeLogger(list):
    def append(self, item):
        super().append(item)
        st.write(item)  # 리스트에 추가됨과 동시에 status 창에 즉시 출력

# 소요 시간 측정용 헬퍼 (함수 내부에 넣거나 직접 호출)
def tic(): return time.perf_counter()
def toc(start): return f"({time.perf_counter() - start:.2f}s)"

# ==========================================
# 1. 초기화 및 환경 설정
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI BuildGraph RAG Engine", layout="wide", page_icon="🏗️")

# ==========================================
# 2. RAG 기반 수석 건축가 추론 엔진 (LLM)
# ==========================================
def analyze_floorplan_with_llm(api_key, logical_graph):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    
    system_prompt = """
    당신은 20년 경력의 수석 건축 설계사입니다. 
    기하학적 연산을 통해 완벽하게 추출된 [위상 구조 지식 그래프(JSON)]를 제공받습니다.
    이 구조는 방(ROOM)이 어떤 벽(WALL)에 둘러싸여(BOUNDED_BY_WALL) 있고, 어떤 문(DOOR)이나 뚫린 입구(OPENING)로 연결되는지 묘사합니다.
    이 데이터를 바탕으로 다음 항목을 분석하여 마크다운 리포트를 작성하세요:
    
    1. 🚶 동선 및 출입 분석: 문(CONNECTED_VIA_DOOR)이나 개구부(CONNECTED_VIA_OPENING)로 연결된 방들의 흐름 분석.
    2. 🤫 프라이버시 및 소음: 벽(ADJACENT_WALL)을 공유하는 방들의 배치가 적절한지 평가.
    3. 🛋️ 공간 활용: 방 내부 빌트인 가구(CONTAINS_FIXTURE)의 배치가 목적에 부합하는지 분석.
    4. ⚠️ 구조적 개선 제안: 건축가 관점에서 도면의 논리적 결함이나 동선 개선점 도출.
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음은 도면의 논리적 지식 그래프입니다. 분석해주세요:\n{json.dumps(logical_graph, ensure_ascii=False)}"}
            ],
            temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM 분석 중 오류 발생: {e}"

# ==========================================
# 3. 텍스트 분류기 및 기하학 파서
# ==========================================
def classify_texts_with_llm(api_key, text_list, debug_logs):
    start_time = time.perf_counter() # <--- 추가
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
    debug_logs.append(f"⏱️ [LLM 분류 완료] {time.perf_counter() - start_time:.2f}s") # <--- 추가
    return class_map, {t['layer'] for t in text_list if class_map.get(t['text']) == 'FIXTURE'}

def process_entity_advanced(entity, eff_layer, fixture_layers, fixture_polygons, min_door_radius):
    try:
        eff_layer_up = eff_layer.upper()
        o_type = 'WALL'
        
        # 1️⃣ 레이어 기반 우선 분류
        is_door_layer = any(w in eff_layer_up for w in ['DOOR', '문'])
        is_window_layer = any(w in eff_layer_up for w in ['WIND', '창문', '창호'])
        is_fixture_layer = eff_layer in fixture_layers or any(w in eff_layer_up for w in ['FUR', 'FIX', 'WARD', 'CB', 'SINK', '가구'])
        
        if is_door_layer: o_type = 'DOOR'
        elif is_window_layer: o_type = 'WINDOW' # 창문 타입 추가
        elif is_fixture_layer: o_type = 'FIXTURE'

        # 2️⃣ 기하학적 특수 처리 (ARC)
        if entity.dxftype() == 'ARC':
            center, radius = entity.dxf.center, entity.dxf.radius
            s, e = entity.dxf.start_angle, entity.dxf.end_angle
            
            # 각도 차이 계산 (0도 통과 케이스 대응)
            angle_diff = (e - s) % 360
            
            # 80~100도 사이면 문으로 간주
            if 80 <= angle_diff <= 100 and radius >= min_door_radius:
                o_type = 'DOOR'
                p1 = (center.x + radius * math.cos(math.radians(s)), center.y + radius * math.sin(math.radians(s)))
                p2 = (center.x + radius * math.cos(math.radians(e)), center.y + radius * math.sin(math.radians(e)))
                
                # 💡 핵심: 문의 'Core'는 L자가 아니라 양 끝점을 잇는 직선(Chord)으로 설정하는 것이 
                # 나중에 방을 나눌 때(Shapely 연산) 훨씬 안정적입니다.
                chord_line = LineString([p1, p2])
                visual_path = LineString([p1, (center.x, center.y), p2]) # 시각화용 L자
                return visual_path, chord_line, o_type 
            else:
                # 문 레이어가 아닌데 90도 원호가 아니면 가구(변기, 세면대 등)일 확률이 높음
                o_type = 'WINDOW' if is_window_layer else 'FIXTURE'

        # 3️⃣ 일반 선형 엔티티 처리
        p = path.make_path(entity)
        vs = list(p.flattening(distance=0.1))
        if len(vs) < 2: return None, None, o_type
        geom = LineString([(v.x, v.y) for v in vs])

        # 4️⃣ 교차 검증: 문 레이어의 직선이 벽으로 변하는 것 방지
        # 가구 폴리곤 안에 있으면 무조건 가구로 강제
        for f_poly in fixture_polygons:
            if f_poly.contains(geom):
                o_type = 'FIXTURE'
                break
        
        # 만약 문 레이어인데 여기까지 WALL로 살아남았다면, 문짝이나 문틀이므로 DOOR 유지
        if is_door_layer: o_type = 'DOOR'

        return geom, geom, o_type
    except:
        return None, None, 'WALL'

def calculate_dynamic_tolerances(door_lines, wall_lines, debug_logs):
    door_lengths = sorted([d.length for d in door_lines if d.length > 0.1])
    if door_lengths:
        base_scale = door_lengths[len(door_lengths) // 2] / 2.0
        anchor_type = "문(Door) 실제 폭(반지름)"
    else:
        wall_lengths = sorted([w.length for w in wall_lines if w.length > 0.1])
        idx = max(0, int(len(wall_lengths) * 0.1))
        base_scale = wall_lengths[idx] if wall_lengths else 10.0
        anchor_type = "벽(Wall) 하위 10% 중간값"
    
    tols = {
        "scale": base_scale, "wall_touch": base_scale * 0.15,
        "door_search": base_scale * 0.3, "room_buffer": base_scale * 0.2,
        "open_path_area": (base_scale ** 2) * 0.2
    }
    debug_logs.append(f"\n📏 [Dynamic Scaling] 기준 축척({anchor_type}): {base_scale:.2f}")
    return tols

# ==========================================
# 4. 코어 물리 엔진 (Grow & Prune 독립화 및 Top-2 거리 할당)
# ==========================================
def calculate_rooms(_objects, _text_data, class_map, debug_logs, debris_factor):
    t_physics = time.perf_counter()
    debug_logs.append("\n🧱 [공간 구획 엔진: Grow & Prune]")
    wall_lines = [obj['core_line'] for obj in _objects.values() if obj['type'] == 'WALL']
    door_lines = [obj['core_line'] for obj in _objects.values() if obj['type'] == 'DOOR']
    all_fixtures = {obj['id']: obj['core_line'] for obj in _objects.values() if obj['type'] == 'FIXTURE'}
    tols = calculate_dynamic_tolerances(door_lines, wall_lines, debug_logs)
    dynamic_debris_area = (tols["scale"] ** 2) * debris_factor
    
    # 💡 [핵심 교정] 가벽을 세울 때 문(Door)을 무시하여 충돌(Blocking) 방지
    promoted_fids = set()
    growing = True
    iteration = 0
    while growing:
        iteration += 1
        growing = False
        debug_logs.append(f"  🔄 Grow Iteration {iteration}: 가구 승격 검토 중...")
        current_walls = wall_lines + [all_fixtures[fid] for fid in promoted_fids]
        current_network = current_walls + door_lines
        wall_tree = STRtree(current_walls)
        for fid, f_geom in all_fixtures.items():
            nearby_indices = wall_tree.query(f_geom.buffer(tols["wall_touch"]))
            if fid in promoted_fids: continue
            
            # 오직 벽의 끝점(Boundary)과 문(Door)에 닿지 않은 가구만 승격
            touches_wall_dead_end = False
            for idx in nearby_indices:
                w_line = current_walls[idx]
                b = w_line.boundary
                w_pts = [b] if b.geom_type == 'Point' else (list(b.geoms) if b.geom_type == 'MultiPoint' else [])
                for pt in w_pts:
                    # 벽의 끝점이 다른 네트워크와 연결되지 않았다면 Dead End
                    if not any(other.distance(pt) < tols["wall_touch"] for other in current_network if other != w_line):
                        if f_geom.distance(pt) < tols["wall_touch"]:
                            touches_wall_dead_end = True
                            break
                if touches_wall_dead_end:
                    debug_logs.append(f"    ➕ {fid} 승격됨 (벽에 닿음)") # 어떤 가구가 범인인지 확인
                    promoted_fids.add(fid) 
                    growing = True
                    break
                
            if touches_wall_dead_end:
                promoted_fids.add(fid); growing = True; break
                
    pruning = True
    p_iteration = 0
    while pruning:
        p_iteration += 1
        pruning = False
        debug_logs.append(f"  ✂️ Prune Iteration {p_iteration}: 가벽 유효성 검사 중...")
        current_network = wall_lines + door_lines + [all_fixtures[fid] for fid in promoted_fids]
        for fid in list(promoted_fids):
            f_geom = all_fixtures[fid]
            f_eps = f_geom.boundary
            f_pts = [f_eps] if f_eps.geom_type == 'Point' else (list(f_eps.geoms) if f_eps.geom_type == 'MultiPoint' else [])
            
            # 승격된 가벽 스스로가 허공에 떠있으면 강등
            has_dead_end = False
            for pt in f_pts:
                if not any(other.distance(pt) < tols["wall_touch"] * 0.5 for other in current_network if other != f_geom):
                    has_dead_end = True; break
                    
            if has_dead_end:
                promoted_fids.remove(fid); pruning = True; break

    inferred_walls = []
    for count, fid in enumerate(promoted_fids):
        new_id = f"INFERRED_WALL_{count}"
        _objects[new_id] = {'geom': _objects[fid]['geom'], 'core_line': all_fixtures[fid], 'layer': 'AUTO_INFERRED_WALL', 'type': 'WALL', 'id': new_id}
        inferred_walls.append(all_fixtures[fid])

    final_wall_lines = wall_lines + inferred_walls
    room_barriers = final_wall_lines + door_lines 
    if not room_barriers: return []
    debug_logs.append(f"⏱️ [Grow & Prune 완료] {time.perf_counter() - t_physics:.2f}s") # <--- 추가
    t_topo = time.perf_counter()

    topo = unary_union(room_barriers).buffer(tols["room_buffer"], quad_segs=1, cap_style=3)
    debug_logs.append(f"⏱️ [위상 위원회 구성 완료] {time.perf_counter() - t_topo:.2f}s") # <--- 추가
    raw_spaces = [Polygon(interior) for p in (list(topo.geoms) if hasattr(topo, 'geoms') else [topo]) for interior in p.interiors]
    physical_spaces = [s.buffer(tols["room_buffer"], join_style=2).simplify(tols["scale"] * 0.05) for s in raw_spaces if s.buffer(tols["room_buffer"], join_style=2).simplify(tols["scale"] * 0.05).area > (tols["scale"] ** 2) * 0.1]

    # --- Semantic Naming ---
    room_anchors = [{'point': Point(t['pos']), 'name': t['text']} for t in _text_data if class_map.get(t['text']) == 'ROOM']
    temp_rooms, unnamed = [], []
    for space in physical_spaces:
        r_anchors = [a for a in room_anchors if space.contains(a['point'])]
        if r_anchors:
            names = sorted(list(set([a['name'] for a in r_anchors])))
            temp_rooms.append({'geom': space, 'name': " / ".join(names), 'type': 'NAMED'})
        else: unnamed.append(space)

    final_rooms = []
    name_counts = {}
    for r in temp_rooms:
        safe_name = r['name'].replace('\n', ' ').strip()
        name_counts[safe_name] = name_counts.get(safe_name, 0) + 1
        r['id'] = f"{safe_name} #{name_counts[safe_name]}"
        final_rooms.append(r)

    for j, shard in enumerate(unnamed):
        if shard.area < dynamic_debris_area:
            best_r, max_overlap = None, -1
            buffered_s = shard.buffer(tols["scale"] * 0.1)
            for r in final_rooms:
                overlap = buffered_s.intersection(r['geom']).area
                if overlap > max_overlap: max_overlap, best_r = overlap, r
            if best_r and max_overlap > 0:
                best_r['geom'] = unary_union([best_r['geom'], shard]).simplify(tols["scale"] * 0.05)
                continue
        final_rooms.append({'id': f"UNKNOWN_SPACE #{j}", 'geom': shard, 'name': 'UNKNOWN_SPACE', 'type': 'UNKNOWN'})

    fixture_lines_original = [obj['core_line'] for obj in _objects.values() if obj['type'] == 'FIXTURE']
    fixture_anchors = [{'point': Point(t['pos']), 'name': t['text']} for t in _text_data if class_map.get(t['text']) == 'FIXTURE']
    fixture_union = unary_union(fixture_lines_original).buffer(tols["scale"] * 0.1) if fixture_lines_original else Polygon()
    if fixture_lines_original:
        fixture_polys = list(polygonize(fixture_union))
        for i, f_poly in enumerate(fixture_polys):
            if f_poly.area < (tols["scale"] ** 2) * 0.05: continue
            f_anchors = [a for a in fixture_anchors if f_poly.contains(a['point'])]
            name = " / ".join(sorted(list(set([a['name'] for a in f_anchors])))) if f_anchors else "Built-in Fixture"
            final_rooms.append({'id': f"FIXTURE_ZONE_{i}", 'geom': f_poly, 'name': name, 'type': 'FIXTURE_SPACE'})

    # ---------------------------------------------------------
    # ✂️ [STEP 6: 거리 기반 Top-2 문 할당 및 위상 경계선 추출]
    # ---------------------------------------------------------
    debug_logs.append("\n📐 [위상 기하학 추론] 안쪽 면 분할 및 거리 기반 문 할당 중...")
    wall_union = unary_union(final_wall_lines) if final_wall_lines else Polygon()
    named_rooms = [r for r in final_rooms if r['type'] == 'NAMED']
    
    for r in named_rooms:
        r['bounding_walls'], r['bounding_doors'], r['doors_to'], r['openings_to'], r['adjacent_to'] = [], [], [], [], []
        
        # 1. 벽(WALL) 할당: 방 외곽선과 교차(Intersection)하는 안쪽 면만 추출
        try: room_perimeter = r['geom'].exterior
        except: continue
        for obj_id, obj in _objects.items():
            if obj['type'] == 'WALL':
                obj_poly = obj['core_line'].buffer(tols["wall_touch"] * 1.5)
                if room_perimeter.intersects(obj_poly):
                    inner_cut = room_perimeter.intersection(obj_poly)
                    if inner_cut.geom_type in ['LineString', 'MultiLineString'] and inner_cut.length > tols["scale"] * 0.1:
                        r['bounding_walls'].append({'id': obj_id, 'geom': inner_cut})

    # 💡 2. 문(DOOR) 할당: 거리(Distance)를 측정하여 가장 가까운 2개 방에 무조건 할당
    debug_logs.append("\n🚪 [문(Door) 최단거리 자동 할당]")
    door_objs = {k: v for k, v in _objects.items() if v['type'] == 'DOOR'}
    for d_id, d_obj in door_objs.items():
        d_geom = d_obj['core_line']
        room_distances = []
        for r in named_rooms:
            try:
                dist = r['geom'].distance(d_geom)
                room_distances.append({'room': r, 'dist': dist})
            except: pass
        
        # 거리가 가까운 순으로 정렬하여 상위 2개 방 추출
        room_distances.sort(key=lambda x: x['dist'])
        
        # 허용 범위(Scale * 3.0) 이내에 있는 방만 유효 처리
        valid_rooms = [x for x in room_distances if x['dist'] < tols["scale"] * 3.0][:2]
        
        debug_logs.append(f"  ▶ {d_id} -> 할당된 방: {', '.join([vr['room']['id'] for vr in valid_rooms])}")
        
        for vr in valid_rooms:
            # 시각화를 위해 원래 문의 형태(L자)를 그대로 전달
            vr['room']['bounding_doors'].append({'id': d_id, 'geom': d_geom})

    # 3. 위상 관계망(Topology) 매핑
    debug_logs.append("\n🔍 [위상 관계망(Topology) 심층 분석 시작]")
    for i, r1 in enumerate(named_rooms):
        r1_doors = {d['id'] for d in r1['bounding_doors']}
        for j, r2 in enumerate(named_rooms): 
            if i >= j: continue
            
            # [A] 출입문 연결
            shared_doors = r1_doors.intersection({d['id'] for d in r2['bounding_doors']})
            if shared_doors:
                door_id = list(shared_doors)[0]
                r1['doors_to'].append((r2['id'], door_id))
                r2['doors_to'].append((r1['id'], door_id))
                debug_logs.append(f"  ✅ [문 연결] {r1['id']} ↔ {r2['id']} | 공유 문: {door_id}")
                continue
            
            # [B] 인접성 및 개구부 통로
            try:
                r1_bound = r1['geom'].exterior.buffer(tols["wall_touch"] * 2)
                r2_bound = r2['geom'].exterior.buffer(tols["wall_touch"] * 2)
            except: continue

            if r1_bound.intersects(r2_bound):
                shared_area = r1_bound.intersection(r2_bound)
                open_path = shared_area.difference(wall_union.buffer(tols["wall_touch"]))
                open_path_val = round(open_path.area if not open_path.is_empty else 0, 2)
                threshold_val = round(tols["open_path_area"], 2)
                
                if open_path_val > threshold_val:
                    r1['openings_to'].append(r2['id'])
                    r2['openings_to'].append(r1['id'])
                    debug_logs.append(f"  🌬️ [개구부 연결] {r1['id']} ↔ {r2['id']} | 통로 면적: {open_path_val} (기준: {threshold_val})")
                else:
                    r1['adjacent_to'].append(r2['id'])
                    r2['adjacent_to'].append(r1['id'])
                    debug_logs.append(f"  🧱 [단순 벽 인접] {r1['id']} ↔ {r2['id']} | 통로 없음 (면적: {open_path_val})")

    return final_rooms

# ==========================================
# 5. 지식 그래프 결정론적 추출 엔진
# ==========================================
def extract_logical_knowledge_graph(rooms, objects_dict, debug_logs):
    t_graph = time.perf_counter()
    debug_logs.append("\n🕸️ [지식 그래프(GraphRAG) 정밀 추출 시작]")
    graph = {"nodes": [], "edges": []}
    named_rooms = [r for r in rooms if r['type'] == 'NAMED']
    fixture_spaces = [r for r in rooms if r['type'] == 'FIXTURE_SPACE']
    
    active_walls, active_doors = set(), set()
    for r in named_rooms:
        active_walls.update([w['id'] for w in r.get('bounding_walls', [])])
        active_doors.update([d['id'] for d in r.get('bounding_doors', [])])

    for r in named_rooms: graph["nodes"].append({"id": r['id'], "label": "ROOM", "name": r['name'], "area": round(r['geom'].area, 2)})
    for f in fixture_spaces: graph["nodes"].append({"id": f['id'], "label": "FIXTURE_ZONE", "name": f['name']})
    for w_id in active_walls: graph["nodes"].append({"id": w_id, "label": "WALL", "name": f"Wall {w_id}"})
    for d_id in active_doors: graph["nodes"].append({"id": d_id, "label": "DOOR", "name": f"Door {d_id}"})

    for r in named_rooms:
        r_id = r['id']
        for w in r.get('bounding_walls', []): graph["edges"].append({"source": r_id, "target": w['id'], "relation": "BOUNDED_BY_WALL"})
        for d in r.get('bounding_doors', []): graph["edges"].append({"source": r_id, "target": d['id'], "relation": "HAS_DOOR"})
        for target_id, d_id in r.get('doors_to', []): graph["edges"].append({"source": r_id, "target": target_id, "relation": "CONNECTED_VIA_DOOR", "via": d_id})
        for target_id in r.get('openings_to', []): graph["edges"].append({"source": r_id, "target": target_id, "relation": "CONNECTED_VIA_OPENING"})
        for target_id in r.get('adjacent_to', []): graph["edges"].append({"source": r_id, "target": target_id, "relation": "ADJACENT_WALL"})
        for f in fixture_spaces:
            if r['geom'].intersects(f['geom']) and r['geom'].intersection(f['geom']).area > 0.1:
                graph["edges"].append({"source": r_id, "target": f['id'], "relation": "CONTAINS_FIXTURE"})
                
    debug_logs.append(f"🏁 추출 완료! 노드 {len(graph['nodes'])}개, 엣지 {len(graph['edges'])}개 확보.")
    debug_logs.append(f"⏱️ [지식 그래프 생성 완료] {time.perf_counter() - t_graph:.2f}s") # <--- 추가
    return graph

# ==========================================
# 6. 메인 UI (Streamlit)
# ==========================================
st.sidebar.title("🛠️ AI BuildGraph Ultimate")
uploaded_file = st.sidebar.file_uploader("1️⃣ DXF 도면 업로드", type=['dxf'])

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ 시각화 제어")
show_rooms = st.sidebar.checkbox("방 영역 표시", value=True)
show_unknown = st.sidebar.checkbox("이름 없는 공간/자투리", value=False)
show_walls = st.sidebar.checkbox("벽/문 표시", value=True)
show_fixtures = st.sidebar.checkbox("가구 표시", value=True)
show_knowledge_graph = st.sidebar.checkbox("🕸️ 지식 그래프 네트워크", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 물리 엔진 설정")
debris_factor = st.sidebar.slider("자투리 흡수 배율 (동적 스케일 기반)", 0.1, 2.0, 0.5, 0.1)
debug_inferred_walls = st.sidebar.checkbox("🚨 복원된 가벽 강조", value=False)

if uploaded_file:
    cache_key = f"{uploaded_file.name}_{debris_factor}"
    if 'current_cache_key' not in st.session_state or st.session_state.current_cache_key != cache_key:
        with st.status("🛠️ 공간 데이터 파이프라인 가동 중...", expanded=True) as status:
            debug_logs = RealTimeLogger() # 커스텀 로거 객체 생성
            debug_logs.append("🚀 [엔진 가동] 파이프라인 시작") # 이제 실시간으로 출력됩니다.
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            #debug_logs = ["\n🚀 [엔진 가동] 파이프라인 시작"]
            doc = ezdxf.readfile(tmp_path)
            msp = doc.modelspace()
            os.remove(tmp_path)
            
            txts = [{'text': re.sub(r'\\[a-zA-Z0-9,-]+[^;]*;|[{} ]', '', ent.plain_text() if hasattr(ent, 'plain_text') else ent.dxf.text).strip(), 'pos': (ent.dxf.insert.x, ent.dxf.insert.y), 'layer': ent.dxf.layer} for ent in msp if ent.dxftype() in ('TEXT', 'MTEXT') and re.search(r'[a-zA-Z0-9]', re.sub(r'\\[a-zA-Z0-9,-]+[^;]*;|[{} ]', '', ent.plain_text() if hasattr(ent, 'plain_text') else ent.dxf.text).strip())]
            class_map, fixture_layers = classify_texts_with_llm(OPENAI_API_KEY, txts, debug_logs)
            
            arc_radii = []
            for ent in msp:
                if ent.dxftype() == 'ARC':
                    diff = abs(ent.dxf.end_angle - ent.dxf.start_angle)
                    if diff > 360: diff -= 360
                    if 80 <= diff <= 100: arc_radii.append(ent.dxf.radius)
            
            door_standard = 1.0
            if arc_radii:
                arc_radii.sort()
                door_standard = arc_radii[int(len(arc_radii) * 0.90)]
            min_door_radius = door_standard * 0.3 
            debug_logs.append(f"📏 [ARC 크기 검증] 상위 10% 문 반경: {door_standard:.2f} -> 최소 허용 규격: {min_door_radius:.2f}")
            
            all_raw_lines = []
            for ent in msp:
                if ent.dxftype() not in ('TEXT', 'MTEXT'):
                    for sub in (ent.virtual_entities() if ent.dxftype() == 'INSERT' else [ent]):
                        try:
                            vp = list(path.make_path(sub).flattening(distance=0.1))
                            if len(vp) >= 2: all_raw_lines.append(LineString([(v.x, v.y) for v in vp]))
                        except: pass
            t_geo = time.perf_counter()
            all_raw_polys = sorted(list(polygonize(unary_union(all_raw_lines))), key=lambda x: x.area)
            debug_logs.append(f"⏱️ [기하 구조화 완료] {time.perf_counter() - t_geo:.2f}s") # <--- 추가
            fixture_polygons = [poly.buffer(0.5) for t_dict in txts if class_map.get(t_dict['text']) == 'FIXTURE' for poly in all_raw_polys if poly.contains(Point(t_dict['pos']))]

            objs = {}
            type_counters = {'WALL': 0, 'FIXTURE': 0, 'DOOR': 0, 'UNKNOWN': 0}
            
            for ent in msp:
                if ent.dxftype() not in ('TEXT', 'MTEXT'):
                    for sub in (ent.virtual_entities() if ent.dxftype() == 'INSERT' else [ent]):
                        eff_layer = ent.dxf.layer if sub.dxf.layer == '0' else sub.dxf.layer
                        geom_visual, core_line, o_type = process_entity_advanced(sub, eff_layer, fixture_layers, fixture_polygons, min_door_radius)
                        if geom_visual:
                            type_counters[o_type] = type_counters.get(o_type, 0) + 1
                            new_id = f"{o_type}_{type_counters[o_type]}"
                            objs[new_id] = {'geom': geom_visual, 'core_line': core_line, 'layer': eff_layer, 'type': o_type, 'id': new_id}
            
            rooms = calculate_rooms(objs, txts, class_map, debug_logs, debris_factor)
            logical_graph = extract_logical_knowledge_graph(rooms, objs, debug_logs)

            status.update(label="🎉 도면 구조 분석 완료!", state="complete", expanded=False)
            st.session_state.update({
                'objects': objs, 'texts': txts, 'class_map': class_map, 
                'rooms': rooms, 'logical_graph': logical_graph, 
                'debug_logs': debug_logs, 'current_cache_key': cache_key,
                'llm_analysis': None
            })
        #elapsed = time.perf_counter() - t_start
        #st.toast(f"물리 엔진 계산 완료: {elapsed:.2f}초")


    # --- 시각화 섹션 ---
    if 'rooms' in st.session_state and st.session_state.rooms:
        named_room_options = ["전체 보기"] + [r['id'] for r in st.session_state.rooms if r['type'] == 'NAMED']
        selected_room_filter = st.sidebar.selectbox("🔎 특정 방 정밀 안쪽 면(Inner Boundary) 확인", named_room_options)
        
        focus_room = None
        if selected_room_filter != "전체 보기":
            for r in st.session_state.rooms:
                if r['id'] == selected_room_filter: 
                    focus_room = r; break

        fig = go.Figure()
        palette = pcolors.qualitative.Pastel

        for i, r in enumerate(st.session_state.rooms):
            if r['type'] == 'FIXTURE_SPACE': continue
            if r['type'] == 'UNKNOWN' and not show_unknown: continue
            if r['type'] == 'NAMED' and not show_rooms: continue
            
            fill_color = palette[i % len(palette)] if r['type'] == 'NAMED' else 'rgba(150, 150, 150, 0.3)'
            opacity = 0.8 if not focus_room or r['id'] == focus_room['id'] else 0.1
            
            hover_info = f"<b>{r['name']}</b><br>ID: {r['id']}"
            if r['type'] == 'NAMED':
                hover_info += f"<br>안쪽 벽: {len(r.get('bounding_walls', []))}개"
                hover_info += f"<br>연결 문: {len(r.get('bounding_doors', []))}개"
            
            for poly in (list(r['geom'].geoms) if r['geom'].geom_type in ('MultiPolygon', 'GeometryCollection') else [r['geom']]):
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.coords.xy
                    fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='toself', fillcolor=fill_color, line=dict(width=0), name=r['name'], opacity=opacity, hoverinfo="text", hovertext=hover_info))
            if r['type'] == 'NAMED': fig.add_annotation(x=r['geom'].centroid.x, y=r['geom'].centroid.y, text=f"<b>{r['name']}</b>", showarrow=False, font=dict(size=10, color="black"))

        for i, r in enumerate(st.session_state.rooms):
            if r['type'] != 'FIXTURE_SPACE' or not show_fixtures: continue
            for poly in (list(r['geom'].geoms) if r['geom'].geom_type in ('MultiPolygon', 'GeometryCollection') else [r['geom']]):
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.coords.xy
                    fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='toself', fillcolor='rgba(243, 156, 18, 0.7)', line=dict(width=1, color='#d35400'), name=r['name'], opacity=1.0))

        for obj_id, obj in st.session_state.objects.items():
            otype = obj['type']
            if (otype in ('WALL', 'DOOR') and not show_walls) or (otype == 'FIXTURE' and not show_fixtures): continue
            
            color = '#2c3e50'
            width = 2
            opacity = 0.2 if focus_room else 1.0
            
            if otype == 'DOOR': color = '#2980b9'
            elif otype == 'FIXTURE': color = '#e67e22'
            elif otype == 'WALL' and debug_inferred_walls and obj.get('layer') == 'AUTO_INFERRED_WALL': color = '#ff0000'; width = 4
                    
            x, y = obj['geom'].exterior.coords.xy if hasattr(obj['geom'], 'exterior') else obj['geom'].xy
            fig.add_trace(go.Scatter(x=list(x)+[None], y=list(y)+[None], fill='none', line=dict(color=color, width=width), opacity=opacity, showlegend=False, hoverinfo="text", hovertext=f"{otype}: {obj_id}"))

        if focus_room:
            for w in focus_room.get('bounding_walls', []):
                geoms = w['geom'].geoms if w['geom'].geom_type in ('MultiLineString', 'GeometryCollection') else [w['geom']]
                for g in geoms:
                    if g.geom_type == 'LineString':
                        x, y = g.xy
                        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='#e74c3c', width=6), showlegend=False, hoverinfo="text", hovertext=f"잘린 안쪽 벽: {w['id']}"))
            for d in focus_room.get('bounding_doors', []):
                geoms = d['geom'].geoms if d['geom'].geom_type in ('MultiLineString', 'GeometryCollection') else [d['geom']]
                for g in geoms:
                    if g.geom_type == 'LineString':
                        x, y = g.xy
                        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='#9b59b6', width=6), showlegend=False, hoverinfo="text", hovertext=f"할당된 문: {d['id']}"))

        if show_knowledge_graph and 'logical_graph' in st.session_state and st.session_state.logical_graph:
            node_coords = {}
            for r in st.session_state.rooms:
                if r['type'] in ['NAMED', 'FIXTURE_SPACE']: node_coords[r['id']] = (r['geom'].centroid.x, r['geom'].centroid.y)
            for obj_id, obj in st.session_state.objects.items():
                if obj['type'] in ['WALL', 'DOOR']:
                    try: node_coords[obj_id] = (obj['geom'].centroid.x, obj['geom'].centroid.y)
                    except: pass
            
            edge_styles = {
                "CONNECTED_VIA_DOOR": dict(color="rgba(46, 204, 113, 0.9)", width=3, dash="dash"),
                "CONNECTED_VIA_OPENING": dict(color="rgba(52, 152, 219, 0.8)", width=2, dash="dashdot"),
                "ADJACENT_WALL": dict(color="rgba(231, 76, 60, 0.5)", width=1, dash="dot"),
            }
            
            for edge in st.session_state.logical_graph.get("edges", []):
                s_id, t_id, rel = edge["source"], edge["target"], edge["relation"]
                if rel in ["BOUNDED_BY_WALL", "HAS_DOOR", "CONTAINS_FIXTURE"]: continue
                if s_id in node_coords and t_id in node_coords:
                    x0, y0 = node_coords[s_id]; x1, y1 = node_coords[t_id]
                    fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode="lines", line=edge_styles.get(rel, dict(color="black", width=1)), hoverinfo="text", hovertext=f"[{rel}]<br>{s_id} ↔ {t_id}", showlegend=False))

            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in st.session_state.logical_graph.get("nodes", []):
                n_id, label, name = node["id"], node["label"], node["name"]
                if label == 'WALL': continue
                if n_id in node_coords:
                    nx, ny = node_coords[n_id]
                    node_x.append(nx); node_y.append(ny)
                    if label == 'ROOM': node_color.append('#3498db'); node_size.append(14)
                    elif label == 'FIXTURE_ZONE': node_color.append('#f39c12'); node_size.append(8)
                    elif label == 'DOOR': node_color.append('#9b59b6'); node_size.append(10)
                    node_text.append(f"<b>{name}</b><br>ID: {n_id}<br>Type: {label}")
                    
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')), hoverinfo="text", hovertext=node_text, showlegend=False))

        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1, visible=False), xaxis=dict(visible=False), plot_bgcolor='white', height=700, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        tab_ai, tab_json, tab_log = st.tabs(["💡 AI 수석 건축가 리포트", "🕸️ 지식 그래프 (JSON)", "🕵️‍♂️ 디버그 로그"])

        with tab_ai:
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🚀 인사이트 생성", type="primary"):
                    with st.spinner("논리적 구조를 분석 중입니다..."):
                        st.session_state.llm_analysis = analyze_floorplan_with_llm(OPENAI_API_KEY, st.session_state.logical_graph)
            with col2:
                if st.session_state.get('llm_analysis'):
                    st.markdown(st.session_state.llm_analysis)
                else:
                    st.info("왼쪽 버튼을 눌러 도면의 논리적 구조망을 기반으로 한 건축적 통찰을 받아보세요.")

        with tab_json:
            st.caption("LLM 또는 Vector DB에 직접 주입(Injection)할 수 있는 확정적 공간 노드/엣지 데이터입니다.")
            st.json(st.session_state.logical_graph)

        with tab_log:
            st.code("\n".join(st.session_state.debug_logs))