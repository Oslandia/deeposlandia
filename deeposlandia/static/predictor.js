// Load a new image from the static directory on click
var predict_button = document.getElementById("predict_button");
predict_button.addEventListener("click", function(){
  var uploaded_image = document.getElementById("uploaded_image");

  console.log("Predict labels...");  
  document.getElementById("result_label").innerHTML = "";

  var filename = uploaded_image.src;
  predict_labels(filename, "mapillary", "feature_detection");

  console.log("Prediction OK!");
  function predict_labels(filename, dataset, model){
    $.getJSON('/_prediction', {
      img: filename,
      dataset: dataset,
      model: model
    }, function(data){
      var result = [];
      console.log("Pre-loop log");
      $.each(data, function(image, predictions){
  	result.push("<ul>");
  	$.each(predictions, function(key, val){
      	  result.push("<li>" + key + ": " + val + "%</li>" );
  	});
  	result.push("</ul>");
  	console.log(result.join(""))
      });
      document.getElementById("result_label").innerHTML = result.join("")
    });/*$.getJSON*/
  };

});
