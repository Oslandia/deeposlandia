// Load a new image from the static directory on click
document.getElementById("predict_labels").addEventListener("click", function(){

  document.getElementById("result").innerHTML = "";
  console.log("Randomly pick an image filename");

  get_image_name(dataset);
});

function get_image_name(dataset, filename_callback){
  $.ajax({type: 'GET',
	  url: '/mapillary_image_selector',
	  data: {dataset: dataset},
	  success: function(response){
	    var filename = "/static/" + dataset + "/" + response.image_name;
	    console.log("Generate a new image starting from file " + response.image_name + "...");
	    generate_image(document.getElementById("raw_image"), filename);
	    console.log("Predict labels for " + model);
	    predict_labels(filename, dataset, model);
	    console.log("Prediction OK!");  
	  }
	 });
}
  
function generate_image(image, image_name){
  if(image.complete){
    var new_image = new Image();
    new_image.id = "raw_image";
    new_image.src = image_name
    image.parentNode.insertBefore(new_image, image);
    image.parentNode.removeChild(image);
  }
};

function predict_labels(filename, dataset, model){
  $.getJSON('/demo_prediction', {
    img: filename,
    dataset: dataset,
    model: model
  }, function(data){
    var result = [];
    if (model === "feature_detection") {
      $.each(data, function(image, predictions){
	result.push("<ul>");
	$.each(predictions, function(key, val){
      	  result.push("<li>" + key + ": " + val + "%</li>" );
	});
	result.push("</ul>");
      });
      document.getElementById("result").innerHTML = result.join("")
    } else {
      var predicted_image_path;
      $.each(data.lab_images, function(image, predicted_image){
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
      document.getElementById("result").innerHTML = result.join("");
      document.getElementById("predicted_image").src = predicted_image_path;
    }
  });/*$.getJSON*/
};
