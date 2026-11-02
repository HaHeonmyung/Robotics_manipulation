"""Contact sheet of frames from evaluation videos (python frame_strip.py OUT.png LABEL=video.mp4 ... [--times 0,8,16,24,32,40])."""
import sys, subprocess, imageio_ffmpeg, numpy as np
from PIL import Image, ImageDraw
args=[a for a in sys.argv[1:] if not a.startswith("--")]; out=args[0]; items=[a.split("=",1) for a in args[1:]]
times=[float(t) for t in next((a.split("=",1)[1] for a in sys.argv if a.startswith("--times=")),"0,8,16,24,32,40").split(",")]
ff=imageio_ffmpeg.get_ffmpeg_exe(); rows=[]
for label,path in items:
    frames=[]
    row_times=times
    if "@" in path:
        path,tt=path.rsplit("@",1); row_times=[float(x) for x in tt.split(",")]
    for t in row_times:
        p=subprocess.run([ff,"-loglevel","error","-ss",str(t),"-i",path,"-frames:v","1","-f","image2pipe","-vcodec","png","-"],capture_output=True)
        if p.stdout:
            import io; im=Image.open(io.BytesIO(p.stdout)).convert("RGB")
        else:
            im=Image.new("RGB",(320,240),(40,40,40))
        im=im.resize((320,int(320*im.height/im.width)))
        d=ImageDraw.Draw(im); d.rectangle([0,0,62,16],fill=(0,0,0)); d.text((3,2),f"{t:.0f}s",fill=(255,255,255)); frames.append(im)
    w=sum(f.width for f in frames)+4*(len(frames)-1); h=frames[0].height+22
    row=Image.new("RGB",(w,h),(255,255,255)); ImageDraw.Draw(row).text((2,4),label,fill=(0,0,0)); x=0
    for f in frames: row.paste(f,(x,22)); x+=f.width+4
    rows.append(row)
W=max(r.width for r in rows); H=sum(r.height for r in rows)+8*(len(rows)-1)
sheet=Image.new("RGB",(W,H),(255,255,255)); y=0
for r in rows: sheet.paste(r,(0,y)); y+=r.height+8
sheet.save(out); print("saved",out,sheet.size)
