from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import pickle

from bornholdt import simulation

app = FastAPI()


data_event = asyncio.Event()
shutdown_event = asyncio.Event()

fire_data = None
background_tasks = []

#---------------------neuron positions on page---------------------------------------------
neuron_positions = []
spacing = 40
left_gap = 100
top_gap = 100
layer_id = 0


# 20x10 arrangement for 200 neurons 
grid_W, grid_H = 20,10

for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing,
            'y': top_gap + i * spacing,
            'layer': layer_id
        })


#-------------------------------------------------------------------------------------



@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

class Params(BaseModel):
    u_se_ampa: float
    u_se_nmda: float
    u_se_gaba: float
    tau_rec_ampa: float
    tau_rec_nmda: float
    tau_rec_gaba: float
    tau_rise_ampa: float
    tau_rise_nmda: float
    tau_rise_gaba: float
    learning_rate: float
    weight_scale: float

@app.post("/input_params")
async def get_params(data: Params):
    global fire_data

    print(data)
    # fire_data = await run_in_threadpool(simulation)
    with open("/Users/compneuro1/Documents/projects/SOC/fire_data_hebbian_v.pkl", 'rb') as f:
        fire_data = pickle.load(f)

    data_event.set()
    

    return "finish computing"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()


    while True:
        # await asyncio.wait(
        #     [data_event.wait(), shutdown_event.wait()],
        #     return_when=asyncio.FIRST_COMPLETED,
        # )

        # # If shutdown triggered before data_event, exit early
        # if shutdown_event.is_set():
        #     await websocket.close()
        #     break
        await asyncio.wait(
        [asyncio.create_task(data_event.wait()), asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_event.is_set():
            await websocket.close()
            break


        await websocket.send_json({"neurons": neuron_positions})
        for t in range(fire_data.shape[0]):
            # states = fire_data[t].view(-1).tolist()
            states = fire_data[t].reshape(-1).tolist()

            await websocket.send_json({"frame": t, "states": states})
            await asyncio.sleep(0.05)  # 20 FPS
        
        data_event.clear()


@app.post("/shutdown")
async def shutdown():
    shutdown_event.set()
    return {"status": "shutdown triggered"}