document.getElementById("startBtn").addEventListener("click", function() {
    fetch("/start-training/")
    .then(response => response.json())
    .then(data => {
        if (data.done) {
            document.getElementById("result").innerHTML = 
              "<h3>Hasil Training</h3>" +
              "<p>Accuracy: " + (data.result.accuracy*100).toFixed(2) + "%</p>" +
              "<p>F1 Score: " + data.result.f1.toFixed(2) + "</p>" +
              "<a href='/download/cnn/' class='btn'>Download CNN Model</a><br>" +
              "<a href='/download/xgb/' class='btn'>Download XGBoost Model</a>";
        }
    });

    let interval = setInterval(function(){
        fetch("/progress/")
        .then(response => response.json())
        .then(data => {
            let percent = Math.floor((data.step / data.total) * 100);
            document.getElementById("progressBar").style.width = percent + "%";
            document.getElementById("progressText").innerText = "Progress: " + percent + "%";
            if (percent >= 100) {
                clearInterval(interval);
            }
        });
    }, 1000);
});

ocument.getElementById("stopBtn").onclick = () => fetch("/stop-training/",{method:"POST"});

async function liveProgress(){
    const res = await fetch("/progress/");
    const data = await res.json();
    updateProgressBar(data);
    
    if(data.status === "done"){
        fetch("/get-metrics/").then(r=>r.json()).then(metrics=>{
            // Update scatter 3D
            const scatterData = {datasets:[{label:"Classes", data:metrics.scatter, backgroundColor:['red','green','blue']}]};
            new Chart(document.getElementById('scatter3d-chart').getContext('2d'),{type:'scatter',data:scatterData});
            
            // Update ROC
            metrics.roc.forEach(roc=>{/* plot per class */});
            
            // F1 table
            const tbody = document.querySelector("#f1-table tbody");
            metrics.f1.forEach((score,i)=>{
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${metrics.classes[i]}</td><td>${score}</td>`;
                tbody.appendChild(tr);
            });
        });
    } else setTimeout(liveProgress,1000);
}

