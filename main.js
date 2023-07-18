const ort = require("onnxruntime-web");
console.log("HELLO WORLD");

document.getElementById("data").onclick = function() {
    main();
};
// ======================================================================
// Global Variables
// ======================================================================

// ======================================================================
// Functions
// ======================================================================

function createArray() {
    const array = [parseFloat(document.getElementById("input1").value),parseFloat(document.getElementById("input2").value),
        parseFloat(document.getElementById("input3").value),        parseFloat(document.getElementById("input4").value),
        parseFloat(document.getElementById("input5").value),        parseFloat(document.getElementById("input6").value),
        parseFloat(document.getElementById("input7").value),        parseFloat(document.getElementById("input8").value),
        parseFloat(document.getElementById("input9").value),        parseFloat(document.getElementById("input10").value),
        parseFloat(document.getElementById("input11").value),        parseFloat(document.getElementById("input12").value),
        parseFloat(document.getElementById("input13").value),        parseFloat(document.getElementById("input14").value),
        parseFloat(document.getElementById("input15").value),        parseFloat(document.getElementById("input16").value),
        parseFloat(document.getElementById("input17").value),        parseFloat(document.getElementById("input18").value),
        parseFloat(document.getElementById("input19").value),        parseFloat(document.getElementById("input20").value),
        parseFloat(document.getElementById("input21").value),        parseFloat(document.getElementById("input22").value),
        parseFloat(document.getElementById("input23").value),        parseFloat(document.getElementById("input24").value),
        parseFloat(document.getElementById("input25").value),        parseFloat(document.getElementById("input27").value),
        parseFloat(document.getElementById("input31").value),        parseFloat(document.getElementById("input32").value),
        parseFloat(document.getElementById("input33").value)];
    
    console.log(array);
    const meanArray = [2.33785861e-01, 1.09028036e+02, 1.21249336e+00,
        1.28898513e-01, 1.18473762e-01, 1.25735479e-01, 1.27791357e-01,
        4.10674312e+01, 3.69413106e+01, 3.93067705e+01, 3.90686407e+01,
        4.64190157e-02, 4.24825720e-02, 4.17380566e-02, 4.65273111e-02,
        2.20028270e+01, 3.20967463e-02, 7.25629440e+00, 4.87439146e+00,
        9.43046759e-01, 2.24863472e+00, 1.43017875e+00, 1.27947055e+00,
        5.74480440e-01, 4.62394482e-01, 8.75459714e-01,
        2.58772206e-01,
        5.34002142e-01, 2.07225652e-01];
    const stdDevArray = [1.43373485e-01, 4.82624309e+01, 6.04297604e-01,
          1.27396171e-01, 1.28052842e-01, 1.28489529e-01, 1.29658820e-01,
          3.87496153e+01, 3.82605863e+01, 3.84452704e+01, 3.79988333e+01,
          5.83068056e-02, 6.02578728e-02, 5.88505640e-02, 6.39103105e-02,
          1.73449228e+00, 1.95207485e-02, 2.74085959e+00, 1.79842290e+00,
          3.68977572e-01, 8.13070178e-01, 6.48402511e-01, 5.52498780e-01,
          1.59006328e-01, 1.84797488e-01, 4.35079947e-02,
          4.37963666e-01,
          4.98846443e-01, 4.05321809e-01];

    const subtracted = array.map((element, index) => element - meanArray[index]);
    const divided = subtracted.map((element, index) => element/stdDevArray[index]);
    return divided;
}

async function main() {
    try {
        // create a new session and load the specific model.
        //
        const session = await ort.InferenceSession.create('./warmup-task.onnx');
        
        // prepare inputs. a tensor need its corresponding TypedArray as data
        const array = createArray();
        console.log(array);
        const inputTensor = new ort.Tensor("float32", array, [1,29]);
        // prepare feeds. use model input names as keys.
        const feeds = { input: inputTensor};
        
        const start = Date.now();
        // feed inputs and run
        const results = await session.run(feeds);

        const end = Date.now();
        // read from results
        const predicted = results.output.data;
        document.getElementById('micro-out-div').innerText = `Predicted Heating End Use: ${predicted}`;

    } catch (e) {
        
        document.getElementById('micro-out-div').innerText = `failed to inference ONNX model.`;
    }
}
