from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parents[1]/'outputs'
FPS=15
DURATION=96
X=[[1,2],[3,4]]
WEIGHTS=[[[1,0],[1,1]],[[0,1],[2,0]],[[1,0],[0,2]]]
COLORS=['#5ddbe9','#be9aff','#ffca75']
FONT={n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n) for n in (20,24,28,34,42)}
RESULTS=[[[sum(X[i][k]*w[j][k] for k in range(2)) for j in range(2)] for i in range(2)] for w in WEIGHTS]
assert RESULTS==[[[1,3],[3,7]],[[2,2],[4,6]],[[1,4],[3,8]]]

def txt(d,x,y,s,n=28,c='#edf3ff'):
    d.text((x,y),s,font=FONT[n],fill=c)

def grid(d,x,y,values,title,color,active=None,completed=4,cell=76):
    txt(d,x,y-48,title,28,color)
    for i in range(2):
        for j in range(2):
            on=active and ((active[0]=='row' and i==active[1]) or (active[0]=='col' and j==active[1]) or (active[0]=='cell' and (i,j)==active[1]))
            rect=(x+j*cell,y+i*cell,x+(j+1)*cell-5,y+(i+1)*cell-5)
            d.rounded_rectangle(rect,radius=9,fill='#2b4256' if on else '#1c293d',outline=color if on else '#34455e',width=3 if on else 1)
            value=str(values[i][j]) if i*2+j<completed else '·'
            txt(d,x+j*cell+24,y+i*cell+10,value,34,color if on else '#edf3ff')

def frame(t):
    im=Image.new('RGB',(1280,720),'#101827');d=ImageDraw.Draw(im)
    txt(d,45,20,'TINY GPT  /  MATRIX MULTIPLICATION',20,COLORS[0])
    if t<6:
        txt(d,45,68,'用具体数字，看清 Q / K / V 是怎样算出来的',34)
        grid(d,110,250,X,'X：两行 = 两个 Token',COLORS[0])
        txt(d,430,245,'每个 Token 有两个输入特征',28)
        txt(d,430,300,'每个输出元素 = 一行与一列对应相乘后求和',28)
        txt(d,430,365,'Q = X @ Wq.T   （K、V 同理）',28,COLORS[0])
        txt(d,70,545,'这里展示一个 Batch：X.shape = [T, D] = [2, 2]',28)
        txt(d,70,600,'权重是人为选定的演示数值；bias=False。',24,'#a8b9d0')
    elif t<90:
        index=min(int((t-6)//28),2);u=(t-6)%28
        name='QKV'[index];w=WEIGHTS[index];wt=list(map(list,zip(*w)));result=RESULTS[index];color=COLORS[index]
        txt(d,45,65,f'{name} Projection：'+('先转置权重矩阵' if u<6 else '逐格计算输出矩阵'),34,color)
        txt(d,45,119,f'{name} = X @ W{name.lower()}.T     |     [2, 2] @ [2, 2] → [2, 2]',24,'#a8b9d0')
        if u<6:
            grid(d,100,265,w,'原始权重 W'+name.lower(),color)
            txt(d,410,292,'转置 / Transpose',28)
            p=max(0,min(1,(u-1)/3));p=p*p*(3-2*p)
            x,y,cell=810,265,76
            txt(d,x,y-48,'W'+name.lower()+'.T',28,color)
            for i in range(2):
                for j in range(2):
                    d.rounded_rectangle((x+j*cell,y+i*cell,x+(j+1)*cell-5,y+(i+1)*cell-5),radius=9,fill='#1c293d',outline='#34455e')
            for i in range(2):
                for j in range(2):
                    px=x+((1-p)*j+p*i)*cell+24
                    py=y+((1-p)*i+p*j)*cell+10
                    txt(d,px,py,str(w[i][j]),34,color if i!=j else '#edf3ff')
            txt(d,75,505,'行变成列：W[i, j] 移动到 W.T[j, i]',28)
            txt(d,75,562,'Transpose rearranges entries. It does not change their values.',24,'#a8b9d0')
        else:
            step=min(int((u-6)//5.5),3);part=(u-6)-step*5.5
            i,j=divmod(step,2)
            done=step+(part>=4.2)
            grid(d,75,265,X,'X：输入',color,('row',i))
            txt(d,275,310,'@',42)
            grid(d,365,265,wt,'W'+name.lower()+'.T',color,('col',j))
            txt(d,582,310,'=',42)
            grid(d,685,265,result,name+'：输出',color,('cell',(i,j)),done)
            txt(d,940,260,f'输出位置 [{i}, {j}]',24,color)
            txt(d,940,311,f'Token {i} 的',24)
            txt(d,940,354,f'输出特征 {j}',24)
            a,b=X[i];c,e=w[j]
            terms=[f'{a} × {c}',f'{b} × {e}']
            d.rounded_rectangle((60,455,1220,551),radius=15,fill='#1c293d')
            txt(d,82,480,f'{name}[{i},{j}] =',34,color)
            txt(d,310,480,terms[0],34,color if part<1.8 else '#edf3ff')
            if part>=1.5:
                txt(d,485,480,'+',34)
                txt(d,555,480,terms[1],34,color if part<3.5 else '#edf3ff')
            if part>=3.1:txt(d,735,480,f'= {a*c} + {b*e}',34)
            if part>=4.2:txt(d,1010,480,f'= {result[i][j]}',34,color)
            # Pair the actual factors inside the highlighted row and column.
            k=0 if part<1.8 else 1
            if part<3.5:
                for x,y in [(75+k*76,265+i*76),(365+j*76,265+k*76)]:
                    d.rounded_rectangle((x+2,y+2,x+69,y+69),radius=8,outline='white',width=4)
            txt(d,75,581,'高亮输入行 × 高亮权重列 → 乘积相加 → 填入输出格',24)
            txt(d,75,627,f'Row {i} × column {j} → one scalar at {name}[{i},{j}]',24,'#a8b9d0')
    else:
        txt(d,45,65,'同一个 X，三组权重，得到三个不同的矩阵',34)
        for k,name in enumerate('QKV'):grid(d,130+380*k,280,RESULTS[k],name+' = X @ W'+name.lower()+'.T',COLORS[k])
        txt(d,75,520,'每一行仍对应原来的 Token；变化的是它的特征数值。',28)
        txt(d,75,580,'Q 与 K 的 Token 匹配，是下一步 Q @ K.T 的计算。',28,COLORS[0])
        txt(d,75,635,'Projection transforms features; attention then compares tokens.',24,'#a8b9d0')
    d.rectangle((0,712,int(1280*t/DURATION),720),fill=COLORS[0])
    return im

cmd=['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'qkv_matrix_multiplication.mp4')]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for n in range(FPS*DURATION):p.stdin.write(frame(n/FPS).tobytes())
p.stdin.close()
if p.wait():raise RuntimeError('Video encoding failed')
preview=Image.new('RGB',(1280,720))
for i,t in enumerate([9,16.5,44.5,93]):preview.paste(frame(t).resize((640,360)),((i%2)*640,(i//2)*360))
preview.save(OUT/'qkv_matrix_preview.png')
print(OUT/'qkv_matrix_multiplication.mp4')
