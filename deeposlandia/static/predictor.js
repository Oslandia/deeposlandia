// Load a new image from the static directory on click
document.getElementById("predict_button").addEventListener("click", function(){

  document.getElementById("result_label").innerHTML = "";
  console.log("Predict labels for " + model);
  var uploaded_image = document.getElementById("uploaded_image");
  var filename = uploaded_image.src;
  predict_labels(filename, "mapillary", model);
  console.log("Prediction OK!");

  function predict_labels(filename, dataset, model){
    
    $.getJSON('/prediction', {
      img: filename,
      dataset: dataset,
      model: model
    }, function(data){
      var result = [];
      var predicted_image_path;
      $.each(data.label_images, function(image, predicted_image){
	predicted_image_path = "/static/predicted_images/" + predicted_image;
	result.push("<img id='predicted_image'><label>Predicted labels</label>");
      });
      result.push("<ul>");
      $.each(data.labels, function(label, color){
	if (label != "background") {
    	  result.push("<li><font color='" + color + "'>" + label + "</font></li>" );
	}
      });
      result.push("</ul>");
      document.getElementById("result_label").innerHTML = result.join("");
      document.getElementById("predicted_image").src = predicted_image_path;
    }); //$.getJSON
  };

});
