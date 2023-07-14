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
    return array;
}

async function main() {
    try {
        // create a new session and load the specific model.
        //
        const session = await ort.InferenceSession.create('./warmup-task.onnx');
        
        // prepare inputs. a tensor need its corresponding TypedArray as data
        const array = createArray();
        const inputTensor = new ort.Tensor("float32", array);
        // prepare feeds. use model input names as keys.
        const feeds = { input: inputTensor};
        
        const start = Date.now();
        // feed inputs and run
        const results = await session.run(feeds);

        const end = Date.now();
        // read from results
        const dataC = results.output.data;
        document.write(`data of result tensor 'c': ${dataC} \n inference time: ${end - start} ms`);

    } catch (e) {
        
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}
