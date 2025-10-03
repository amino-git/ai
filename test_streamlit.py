import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time

st.set_page_config(page_title="Snake Game", page_icon="🐍", layout="centered")

# ---------- إعدادات اللعبة ----------
GRID_SIZE = 20  # حجم كل خلية
CELL_COUNT = 20  # عدد الخلايا أفقياً وعمودياً

# ---------- إدارة الحالة ----------
if 'snake' not in st.session_state:
    st.session_state.snake = [(10, 10)]
if 'direction' not in st.session_state:
    st.session_state.direction = 'RIGHT'
if 'food' not in st.session_state:
    st.session_state.food = (5, 5)
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'speed' not in st.session_state:
    st.session_state.speed = 0.3  # مدة التحديث بالثواني

# ---------- وظائف اللعبة ----------
def move_snake():
    head_x, head_y = st.session_state.snake[-1]
    if st.session_state.direction == 'UP':
        head_y -= 1
    elif st.session_state.direction == 'DOWN':
        head_y += 1
    elif st.session_state.direction == 'LEFT':
        head_x -= 1
    elif st.session_state.direction == 'RIGHT':
        head_x += 1
    new_head = (head_x, head_y)
    
    # تحقق من الاصطدام
    if (head_x < 0 or head_x >= CELL_COUNT or head_y < 0 or head_y >= CELL_COUNT 
        or new_head in st.session_state.snake):
        st.session_state.game_over = True
        return
    st.session_state.snake.append(new_head)
    
    # تحقق من أكل الطعام
    if new_head == st.session_state.food:
        st.session_state.score += 1
        st.session_state.food = spawn_food()
        st.session_state.speed = max(0.05, st.session_state.speed - 0.01)  # زيادة السرعة
    else:
        st.session_state.snake.pop(0)

def spawn_food():
    while True:
        pos = (np.random.randint(0, CELL_COUNT), np.random.randint(0, CELL_COUNT))
        if pos not in st.session_state.snake:
            return pos

def draw_game():
    img = Image.new('RGB', (GRID_SIZE*CELL_COUNT, GRID_SIZE*CELL_COUNT), color=(0,0,0))
    draw = ImageDraw.Draw(img)
    
    # رسم الثعبان
    for x, y in st.session_state.snake:
        draw.rectangle([x*GRID_SIZE, y*GRID_SIZE, (x+1)*GRID_SIZE, (y+1)*GRID_SIZE], fill=(0,255,0))
    
    # رسم الطعام
    fx, fy = st.session_state.food
    draw.rectangle([fx*GRID_SIZE, fy*GRID_SIZE, (fx+1)*GRID_SIZE, (fy+1)*GRID_SIZE], fill=(255,0,0))
    
    return img

# ---------- واجهة المستخدم ----------
st.title("🐍 Snake Game - احترافي")

st.subheader(f"Score: {st.session_state.score}")

# أزرار التحكم
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬆️"):
        st.session_state.direction = 'UP'
with col2:
    if st.button("⬅️"):
        st.session_state.direction = 'LEFT'
with col3:
    if st.button("➡️"):
        st.session_state.direction = 'RIGHT'
if st.button("⬇️"):
    st.session_state.direction = 'DOWN'

# رسم اللعبة
img = draw_game()
st.image(img, use_column_width=True)

# التحكم باللعبة
if not st.session_state.game_over:
    move_snake()
    time.sleep(st.session_state.speed)
    st.experimental_rerun()
else:
    st.warning("💀 Game Over!")
    if st.button("إعادة اللعب"):
        st.session_state.snake = [(10, 10)]
        st.session_state.direction = 'RIGHT'
        st.session_state.food = (5, 5)
        st.session_state.score = 0
        st.session_state.game_over = False
        st.session_state.speed = 0.3
        st.experimental_rerun()
