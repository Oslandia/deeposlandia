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
	    var img_folder = "/static/" + dataset + "/images/";
	    var lbl_folder = "/static/" + dataset + "/labels/";
	    var img_filename = img_folder + response.image_name;
	    if (dataset === "mapillary"){
	      img_filename += ".jpg";
	    } else {
	      img_filename += ".png";
	    }
	    var lbl_filename = lbl_folder + response.image_name + ".png"
	    console.log("File " + response.image_name + " will be used as demo...");
	    generate_image("raw_image", img_filename);
	    generate_image("ground_truth", lbl_filename);
	    console.log("Predict labels for " + model);
	    predict_labels(img_filename, dataset, model);
	    console.log("Prediction OK!");  
	  }
	 });
}

function generate_image(image_id, image_name){
  var dom_image = document.getElementById(image_id);
  if(dom_image.complete){
    var new_image = new Image();
    new_image.id = image_id;
    new_image.src = image_name
    dom_image.parentNode.insertBefore(new_image, dom_image);
    dom_image.parentNode.removeChild(dom_image);
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
