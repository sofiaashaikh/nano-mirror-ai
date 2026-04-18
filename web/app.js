const GRID_SIZE = 10;
let inputData = new Array(GRID_SIZE * GRID_SIZE).fill(0);

function createGrid(elementId, isInteractive) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        const cell = document.createElement('div');
        cell.classList.add('cell');
        if (isInteractive) {
            cell.addEventListener('click', () => {
                cell.classList.toggle('active');
                inputData[i] = cell.classList.contains('active') ? 1 : 0;
            });
        }
        container.appendChild(cell);
    }
}

createGrid('inputGrid', true);
createGrid('outputGrid', false);

document.getElementById('runBtn').addEventListener('click', async () => {
    try {
        const session = await ort.InferenceSession.create('../models/task_403.onnx');
        
        const flatArray = new Float32Array(1 * 10 * 30 * 30).fill(0);
        for (let row = 0; row < GRID_SIZE; row++) {
            for (let col = 0; col < GRID_SIZE; col++) {
                if (inputData[row * GRID_SIZE + col] === 1) {
                    const tensorIndex = 2 * (30 * 30) + row * 30 + col;
                    flatArray[tensorIndex] = 1.0;
                }
            }
        }

        const tensor = new ort.Tensor('float32', flatArray, [1, 10, 30, 30]);
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const feeds = {};
        feeds[inputName] = tensor;

        const results = await session.run(feeds);
        const outputFlat = results[outputName].data;

        const outputCells = document.getElementById('outputGrid').children;
        
        for(let i=0; i<outputCells.length; i++) outputCells[i].classList.remove('active');

        for (let row = 0; row < GRID_SIZE; row++) {
            for (let col = 0; col < GRID_SIZE; col++) {
                const tensorIndex = 1 * (30 * 30) + row * 30 + col;
                
                const mirrorCol = 9 - col;
                
                if (outputFlat[tensorIndex] > 0.5) {
                    outputCells[row * GRID_SIZE + mirrorCol].classList.add('active');
                }
            }
        }
    } catch (e) {
        console.error("Error running model:", e);
        alert("Make sure the web app is running on a local server, and task_403.onnx is in the right folder!");
    }
});