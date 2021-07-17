async function run() {
    
    //get dataset
    const csvUrl = "/data/iris.csv";
    const dataset = tf.data.csv(csvUrl, {
        columnConfigs: {
            species: {
                isLabel : true
            }
        }
    });
    
    //prep labels
    const numFeatures = (await dataset.columnNames()).length - 1;
    const numSamples = 150;
    const convertedData = dataset.map(({xs,ys})=>{
        //one hot encoding of ys
        const labels= [
            ys.species=="setosa"?1:0,
            ys.species=="virginica"?1:0,
            ys.species=="versicolor"?1:0
        ]
        
        return {xs:Object.values(xs), ys:Object.values(labels)};
    }).batch(10);
    
    //create model
    const model = tf.sequential();
    model.add(tf.layers.dense({units:5, inputShape: [numFeatures], activation:"sigmoid"}));
    model.add(tf.layers.dense({units:3, activation:"softmax"}));
    
    //compile
    model.compile({loss:"categoricalCrossentropy", optimizer:tf.train.adam(0.06)});
    
    //train
    await model.fitDataset(convertedData, 
                          {epochs:100, 
                          callbacks: {
                              onEpochEnd: async (epoch, logs)=>{
                                  console.log("epoch: "+ epoch, "loss: "+ logs.loss);
                              }
                          }});
    
    //predict
    const testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);
    
    const prediction = model.predict(testVal);
    const pIndex = tf.argMax(prediction, axis=1).dataSync();
            
    const classNames = ["Setosa", "Virginica", "Versicolor"];
            
    // alert(prediction)
    alert(classNames[pIndex])
    
    
}

run();