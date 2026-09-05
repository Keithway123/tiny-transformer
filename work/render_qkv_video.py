from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / 'outputs'
FONT = 'C:/Windows/Fonts/msyh.ttc'
FPS, DURATION = 12, 63
C = ['#64dce7', '#c49cff', '#ffd080']
FONTS = {n: ImageFont.truetype(FONT, n) for n in (20, 24, 28, 32, 39)}

def label(d, x, y, s, size=28, color='#edf3ff'):
    d.text((x, y), s, font=FONTS[size], fill=color)

titles = [
 ('Q / K / V Projection 是什么？', 'One input. Three learned transformations.'),
 ('一个输入，三组独立权重', 'Three linear layers transform the same input.'),
 ('放大 Q Projection 的计算', 'q = x @ Wq.T    (bias=False)'),
 ('换输入，保持权重不变', 'Fixed weights; new input; recomputed activations.'),
 ('匹配分数在 QK 转置时产生', 'Token matching happens at Q @ K.T.'),
 ('用注意力权重对 V 加权', 'Weights select how much value information to use.'),
 ('回到你的 attention.py', 'Parameters are stored. Activations are computed.')]

def frame(t):
    scene = min(int(t // 9), 6)
    u = t % 9
    im = Image.new('RGB', (1280,720), '#101827')
    d = ImageDraw.Draw(im)
    label(d,55,25,'TINY GPT  /  VISUAL LESSON 01',20,C[0])
    label(d,55,74,titles[scene][0],39)
    label(d,55,133,titles[scene][1],24,'#a9bad0')
    def row(y,s,i=0):
        d.rounded_rectangle((60,y,1220,y+70),radius=14,fill='#213149')
        label(d,82,y+15,s,28,C[i])
    if scene==0:
        row(218,'当前练习输入 X.shape = [2, 4, 8]')
        row(310,'2 个样本 / 每个 4 个 Token / 每个 Token 8 个特征',1)
        row(402,'放大一个 Token，用 x = [1, 2] 展示二维计算',2)
        label(d,65,523,'以下权重为人为设定的算术示例，并非训练结果。',28)
        label(d,65,572,'Illustrative weights, not trained model weights.',24,'#a9bad0')
    elif scene in (1,3):
        changed = scene==3 and u>=4
        x='[2, 1]' if changed else '[1, 2]'
        vals=['[2, 3]','[1, 4]','[2, 2]'] if changed else ['[1, 3]','[2, 2]','[1, 4]']
        label(d,65,192,'输入 x = '+x,32)
        for i,(name,w) in enumerate([('q','[[1, 0], [1, 1]]'),('k','[[0, 1], [2, 0]]'),('v','[[1, 0], [0, 2]]')]):
            y=260+100*i
            label(d,65,y,'x',32,C[i])
            d.line((105,y+23,265,y+23),fill=C[i],width=4)
            px=110+145*((u*.5)%1)
            d.ellipse((px-6,y+17,px+6,y+29),fill='white')
            label(d,290,y,'W'+name+' = '+w,28,C[i])
            label(d,835,y,name+' = '+vals[i],32,C[i])
        label(d,65,581,'每条路径：输出 = x @ weight.T',28)
        label(d,65,626,'W 是参数；q/k/v 是当前输入对应的计算结果。',24,'#a9bad0')
    elif scene==2:
        label(d,65,195,'x = [1, 2]      Wq = [[1, 0], [1, 1]]',32,C[0])
        lines=['q[0] = 1 × 1 + 2 × 0 = 1','q[1] = 1 × 1 + 2 × 1 = 3','q = [1, 3]     Shape: [2] → [2]']
        for i,s in enumerate(lines):
            y=277+i*90
            row(y,s,i)
            if i==min(int(u/3),2):d.rounded_rectangle((60,y,1220,y+70),radius=14,outline=C[i],width=3)
        label(d,65,588,'这里只混合一个 Token 的特征，还没有比较 Token。',28)
    elif scene==4:
        label(d,65,191,'Token A: x=[1,2]     Token B: x=[2,1]',28)
        label(d,65,244,'qA=[1,3]     kA=[2,2]     kB=[1,4]',28,C[0])
        row(314,'A 查询 A：[1,3] · [2,2] = 8')
        row(407,'A 查询 B：[1,3] · [1,4] = 13',1)
        label(d,65,519,'每个 Query 与每个 Key 比较 → [B, T, T]',28)
        label(d,65,571,'GPT：缩放 → 屏蔽未来位置 → Softmax',28,C[2])
        label(d,65,620,'8、13 是原始分数；未来位置随后会被屏蔽。',24,'#a9bad0')
    elif scene==5:
        label(d,65,195,'假设当前 Query 可以看到两个 Token：',28)
        label(d,65,246,'weights=[0.25,0.75]（独立示例，非上一幕的结果）',24,C[2])
        row(315,'vA=[1,4]              vB=[2,2]',2)
        row(408,'output = 0.25 × vA + 0.75 × vB = [1.75, 2.50]')
        label(d,65,531,'Q、K 决定权重；V 提供被加权的信息。',28)
        label(d,65,584,'Q/K determine weights; V supplies content.',24,'#a9bad0')
    else:
        lines=['q_projection = nn.Linear(8, 8, bias=False)','q = q_projection(x)','k = k_projection(x)','v = v_projection(x)']
        for i,s in enumerate(lines):label(d,70,205+i*54,s,28,C[max(i-1,0)])
        label(d,65,442,'X/Q/K/V: [2,4,8]     每组 weight: [8,8]',28)
        label(d,65,495,'创建层：随机初始化。训练：优化器更新权重。',28)
        label(d,65,548,'推理：固定权重，根据输入重新计算 Q/K/V。',28)
        label(d,65,611,'注意：K、V 要调用各自的 projection。',24,C[2])
    label(d,1130,26,f'{scene+1} / 7',20,'#a9bad0')
    d.rectangle((0,712,int(1280*t/DURATION),720),fill=C[0])
    return im

cmd=['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'qkv_projection.mp4')]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for n in range(FPS*DURATION):proc.stdin.write(frame(n/FPS).tobytes())
proc.stdin.close()
if proc.wait():raise RuntimeError('ffmpeg failed')
preview=Image.new('RGB',(1280,720))
for i,t in enumerate([12,22,39,57]):preview.paste(frame(t).resize((640,360)),((i%2)*640,(i//2)*360))
preview.save(OUT/'qkv_preview.png')
print(OUT/'qkv_projection.mp4')
