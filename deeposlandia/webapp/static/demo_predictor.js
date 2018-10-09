// Load a new image from the static directory on click
document.getElementById("predict_labels").addEventListener("click", function(){

  console.log("Randomly pick an image filename");
  $.ajax({type: 'GET',
	  url: PREFIX + '/demo_image_selector',
	  data: {dataset: dataset},
	  success: function(response){
	    var url = PREFIX + "/predictor_demo/" + model + "/" + dataset + "/" + response.image_name;
	    $(location).attr('href', url)
	    console.log("File " + response.image_name + " will be used as demo...");
	    console.log("dataset = " + dataset);
	    console.log("model = " + model);
	    console.log("Prediction and URL update OK!");
	  }
	 });
});
