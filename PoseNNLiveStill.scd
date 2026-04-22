
//--VERSION A LOT OF TO AND FOO BETWEEN SERVER AND LANG INTERPRETER

(

var synth, mss, msmp, win, xyslider;

//create fluid dataset
var mpdata = FluidDataSet(s);
var outdata = FluidDataSet(s);

var mpdict;

var adata = 0.dup(34);
var mpbuf = Buffer.alloc(s,34);
	//var mpbuf = Buffer.alloc(s,1434);
var paramsbuf = Buffer.alloc(s,10);

var counter = 0;
var predicting = false;


//function that remove every pair of the array
var removePairs = { |array|
	array.select { |item, index| (index % 4) < 2 }};

//Fluid regressor
	
var mlp = FluidMLPRegressor(s,
	[7],
	activation:FluidMLPRegressor.sigmoid,
	outputActivation:FluidMLPRegressor.sigmoid,
	maxIter: 1000,
	learnRate:0.1,
	batchSize:1,
	validation:0
);



	//CREATE MULTISLIDER

Window.closeAll;
win = Window("Training", Rect(10, 10, 1000, 320))
.front
.alwaysOnTop_(true);


// multislider parameters synth
mss = MultiSliderView(win,Rect(10, 10, 400, 300))
.elasticMode_(1)
.isFilled_(1)
.action_({
	arg ms;
	//the values of the sliders that control the 
	paramsbuf.setn(0,ms.value);	
}).value_(0.5.dup(10));


//multisilder parameters 
msmp = MultiSliderView(win,Rect(410, 10, 400, 300));
//msmp.value_(~adata);
msmp.value_(adata);
msmp.elasticMode_(1);
msmp.isFilled_(1);
msmp.action_({

	if(predicting,

		mlp.predictPoint(mpbuf, paramsbuf, {
			paramsbuf.getn(0, 10,{
				arg prediction;
				defer{
					mss.value_(prediction);
				};
			});
		});
	);

});
msmp.doAction;

// load data file
Button(win,Rect(830,10,100,20))
	.states_([["Load Data File"]])
	.action_{
		// read the JSON file
		var path = "/home/pemb/python/mediapipe/landmarks_output.json";
		var f = File( path, "r");
		mpdict = f.readAllString.parseJSON;
		
	};




//ADD DATA TO THE DATASETS

Button(win,Rect(830,40,100,20))
	.states_([["Add Points"]])
	.action_{
		var id = "point-%".format(counter);

		mpbuf = Buffer.loadCollection(s, mpdict.at("%".format(counter)).at("landmarks").asFloat, 1);
		//		mpdata = FluidDataSet(s).fromBuffer(mpbuf).print;
		
		mpdata.addPoint(id, mpbuf);
		outdata.addPoint(id, paramsbuf);
		counter = counter + 1;

		mpdata.print;
		outdata.print;
	};

//SAVE DATA POINTS COLLECTED TO A FILE

Button(win,Rect(830,70,100,20))
.states_([["Save Data"]])
.action_{
	FileDialog({
		arg folder;
		folder.postln;
		mpdata.write(folder+/+"xydata.json");
		outdata.write(folder+/+"paramsdata.json");
	},{},2,0,true);
	};

//RESTORE DATA POINTS COLLECTED FROM A FILE

Button(win,Rect(830,100,100,20))
.states_([["Load Data"]])
.action_{
	FileDialog({
		arg folder;
		mpdata.read(folder+/+"xydata.json");
		outdata.read(folder+/+"paramsdata.json");
	},fileMode:2,acceptMode:0,stripResult:true);
};

//TRAIN MODEL

Button(win,Rect(830,130,100,20))
.states_([["Train"]])
.action_{
	mlp.fit(mpdata, outdata,{
		arg loss;
		loss.postln;
	});
};

//SAVE TRAINED MODEL

Button(win,Rect(830,160,100,20))
.states_([["Save MLP"]])
.action_{
	Dialog.savePanel({
		arg path;
		if(PathName(path).extension != "json"){
			path = "%.json".format(path);
		};
		mlp.write(path);
	});
};

//LOAD SAVED MODEL

Button(win,Rect(830,190,100,20))
.states_([["Load MLP"]])
.action_{
	Dialog.openPanel({
		arg path;
		mlp.read(path);
	});
};


//INFERENCE OR TRAINING SWITCH

Button(win,Rect(830,220,100,20))
.states_([["Not Predicting"],["Predicting"]])
.action_{
	arg but;
	predicting = but.value.asBoolean;
};

//s.sync;

)




~test = FluidDataSet(s).read("/home/pemb/python/mediapipe/landmarks_output.json").asJSON;


(
{
	// read the JSON file
	var path = "/home/pemb/python/mediapipe/landmarks_output.json";
	var f = File( path, "r");
	~myDict = f.readAllString.parseJSON;
	
}.value();
);

~myDict.at("0").at("landmarks")[0]
~mibuf = Buffer.loadCollection(s, ~myDict.at("0").at("landmarks").asFloat, 1);
~midataset = FluidDataSet(s).fromBuffer(~mibuf).print;



//SYNTH
synth = {

	//define that the dataset control the arguments of the synth.
	var val = FluidBufToKr.kr(paramsbuf); 

	var osc1, osc2, feed1, feed2, base1=69, base2=69, base3 = 130, loud1, loud2;
	#feed2, feed1 = LocalIn.ar(2);

	//only for testing is we get the data from MP. 
	//	FluidBufToKr.kr(~mpbuf).poll;

	loud1 = FluidLoudness.kr(feed1,[\loudness],1,0,hopSize:64);
	loud2 = FluidLoudness.kr(feed2,[\loudness],1,0,hopSize:64);

	osc1 = SinOsc.ar((((feed1 * val[0]) +  val[1]) * base1).midicps,mul:(val[2] * 50).dbamp).atan;
	osc1 = MoogFF.ar(osc1,(base3-(val[3]*(loud2.clip(-120,0)+120))).lag(128/44100).midicps,val[4]*3.5);

	osc2 = SinOsc.ar((((feed2 * val[5]) +  val[6]) * base2).midicps,mul: (val[7] * 50).dbamp).atan;
	osc2 = MoogFF.ar(osc2,(base3-(val[8]*(loud1.clip(-120,0)+120))).lag(128/44100).midicps,val[9]*3.5);

	Out.ar(0,LeakDC.ar([osc1,osc2],mul:0.1));
	LocalOut.ar([osc1,osc2]);
}.play;


)

///////////////////////////////////////////////

//--- VERSION LESS TO AND FOO BETWEEN SERVER AND INTERPRETER

(
var synth, uv, mss, msmp, msdel, sdelta, win, xyslider, mpdict, sldactivity;

var livetraining = false;
//create fluid dataset
var mpdata = FluidDataSet(s);
var outdata = FluidDataSet(s);
// Create array face data

// number of values of the arrays
var numValues = 66;

//var adata = 0.dup(1434);
//var adata = 0.dup(numValues);
//var bdata = 0.dup(numValues);

var adata = Array.fill(numValues, 0);
var bdata = Array.fill(numValues, 0);

// Variables to get the denta change

var padata = Array.fill(numValues, 0);
var deltas = Array.fill(numValues, 0);
var sdeltas = Array.fill(numValues, 0);
var activity = Array.fill(1, 0);


//create the buffers for input and output data
var mpbuf = Buffer.alloc(s, 66);
//var mpbuf = Buffer.alloc(s,1434);
var paramsbuf = Buffer.alloc(s,11);

//delta buffer
var deltabuf = Buffer.alloc(s, 66);


var counter = 0;
//var predicting = false;
var predicting = 0;
var predict = false;

var visibility = true;

var circle = {
	|x, y, r|
	Pen.fillColor = Color.new255(255, 0, 0, 255);
	Pen.strokeColor = Color.red;
	Pen.addArc(x@y + ((r@r)), r, 0, 2pi);
	Pen.fillStroke;
};


//Fluid Regressor Declaration 	
var mlp = FluidMLPRegressor(s,
	[7],
	activation:FluidMLPRegressor.sigmoid,
	outputActivation:FluidMLPRegressor.sigmoid,
	maxIter: 1000,
	learnRate:0.1,
	batchSize:1,
	validation:0
);

//---OSC Message receiver From MediaPipe---//
//-----------------------------------------//

OSCdef(\oscposes, {
	arg msg, mmsg;

	var smoothing = 0.9;	
	// clean the OSC path
	var incarray = msg[1..];

	adata.clear;

	// take out each thris element Z axis
	adata = incarray.reject({
		arg item, i;
		(i + 1) % 3 == 0 }
	);

	mpbuf.setn(0, adata);

	// Calculate the amount of movement and keep it in an array
	adata.do { |val, i|
		deltas[i] = val - padata[i];
		sdeltas[i]  = (smoothing * sdeltas[i]) + ((1 - smoothing) * deltas[i]);
		padata[i] = val;
	};

	// take the absolute and mean of the delta changes
	activity[0] = sdeltas.abs.mean;
	//activity[0].postln;
	
	deltabuf.setn(0, sdeltas);
	
	if(predict == true, {
		{msmp.valueAction_(adata)}.defer;
		//{msdel.valueAction_(sdeltas.linlin(-0.1, 0.1, 0.0, 1.0))}.defer;
		{sdelta.valueAction_(activity.linlin(0.0, 0.1, 0.0, 1.0))}.defer;
		defer{uv.refresh};
		
	});

	if(livetraining == true, {
		{msmp.valueAction_(adata)}.defer;
		//{msdel.valueAction_(sdeltas.linlin(-0.1, 0.1, 0.0, 1.0))}.defer;
		{sdelta.valueAction_(activity.linlin(0.0, 0.1, 0.0, 1.0))}.defer;
		defer{ uv.refresh };
	});

}, '/landmarks/0');

//OSCdef.freeAll;

//---GUI---//
//---------//

//Create window

Window.closeAll;
win = Window("Training Window", Rect(10, 10, 1500, 320))
.front
.alwaysOnTop_(true);

//Multislider Synth arguments 

mss = MultiSliderView(win, Rect(10, 10, 400, 300))
.elasticMode_(1)
.isFilled_(1)
.action_({
	arg ms;
	paramsbuf.setn(0, ms.value); 
}).value_(0.5.dup(11));

//Multislider MediaPipe from OSC/JSON

msmp = MultiSliderView(win, Rect(420, 10, 400, 300));
if(livetraining == true){msmp.value_(adata)};
if((livetraining == false) && (predict == false), {msmp.value_(bdata)});
msmp.elasticMode_(1);
msmp.isFilled_(0);
msmp.drawLines_(0);
msmp.drawRects_(1);
msmp.showIndex = false;
msmp.doAction;

uv = UserView(win, Rect(830, 10, 400, 300));
uv.background_(Color.black);
uv.drawFunc = { |uvview|

	if(visibility == true){
	
	if((livetraining == true)) {
		adata.pairsDo{ |a, b|
			circle.value(
				a * uv.bounds.width,
				b * uv.bounds.height,
				2
			);
		};
	};

	if((predict == true)) {
		adata.pairsDo{ |a, b|
			circle.value(
				a * uv.bounds.width,
				b * uv.bounds.height,
				2
			);
		};
	};
	
	if((livetraining == false) && (predict == false)) {
		bdata.pairsDo{ |a, b|
			circle.value(
				a * uv.bounds.width,
				b * uv.bounds.height,
				2
			);
		};
	};

	};
};


//Multislider DELTA OSC/JSON
/*
msdel = MultiSliderView(win, Rect(810, 10, 400, 300));
msdel.value_(sdeltas.linlin(-0.1, 0.1, 0.0, 1.0));
msdel.elasticMode_(1);
msdel.isFilled_(0);
msdel.doAction; 
	*/
sdelta = MultiSliderView(win, Rect(1240, 10, 20, 300));
//sdelta.size(1);
sdelta.value_(activity.linlin(0.0, 0.001, 0.0, 1.0));
sdelta.elasticMode_(1);
sdelta.isFilled_(1);
sdelta.doAction;

//sldactivity = Slider(win, Rect(1210, 10, 10, 300)).value_(activity);

// load data file
Button(win,Rect(1270,10,100,20))
.states_([["Load Data File"]])
.action_{
	FileDialog({
		arg path, f;

		path.postln;
		f = File( path, "r");
		mpdict = f.readAllString.parseJSON;
		bdata = mpdict.at("%".format(counter)).at("landmarks").asFloat;
		bdata = bdata.reject({ arg item, i; (i + 1) % 3 == 0 });
		{msmp.valueAction_(bdata)}.defer;
		mpbuf.setn(0, bdata);
		defer{ uv.refresh };
	}, {} , 1, 0, true);
};

//ADD DATA TO THE DATASETS

Button(win,Rect(1270,40,100,20))
.states_([["Add Points"]])
.action_{
	var id = "point-%".format(counter);
	
	if(livetraining == false) {
		if(predict == false){
			bdata = mpdict.at("%".format(counter)).at("landmarks").asFloat;
			bdata = bdata.reject({ arg item, i; (i + 1) % 3 == 0 });
			{msmp.valueAction_(bdata)}.defer;
			mpbuf.setn(0, bdata);
			defer{ uv.refresh };
		};
	};
		
	mpdata.addPoint(id, mpbuf);
	outdata.addPoint(id, paramsbuf);
	counter = counter + 1;
	mpdata.print;
	outdata.print;		
};

//SAVE DATA POINTS COLLECTED TO A FILE

Button(win,Rect(1270,70,100,20))
.states_([["Save Data"]])
.action_{
	FileDialog({
		arg folder;
		folder.postln;
		mpdata.write(folder+/+"xydata.json");
		outdata.write(folder+/+"paramsdata.json");
	}, {}, 2, 0, true);
	};

//RESTORE DATA POINTS COLLECTED FROM A FILE

Button(win,Rect(1270,100,100,20))
.states_([["Load Data"]])
.action_{
	FileDialog({
		arg folder;
		mpdata.read(folder+/+"xydata.json");
		outdata.read(folder+/+"paramsdata.json");
	}, fileMode: 2, acceptMode: 0, stripResult: true);
};

//TRAIN MODEL

Button(win,Rect(1270,130,100,20))
.states_([["Train"]])
.action_{
	mlp.fit(mpdata, outdata,{
		arg loss;
		loss.postln;
	});
};

//SAVE TRAINED MODEL

Button(win,Rect(1270, 160, 100, 20))
.states_([["Save MLP"]])
.action_{
	Dialog.savePanel({
		arg path;
		if(PathName(path).extension != "json"){
			path = "%.json".format(path);
		};
		mlp.write(path);
	});
};

//LOAD SAVED MODEL

Button(win,Rect(1270, 190, 100, 20))
.states_([["Load MLP"]])
.action_{
	Dialog.openPanel({
		arg path;
		mlp.read(path);
	});
};


//INFERENCE OR TRAINING SWITCH

Button(win,Rect(1270, 220, 100, 20))
.states_([["Not Predicting"],["Predicting"]])
.action_{
	arg but;
	if(but.value == 1){predict = true}{predict == false};
	synth.set(\predicting, but.value);
};

Button(win,Rect(1270,250,100,20))
.states_([["File Training"],["Live Training"]])
.action_{
	arg msg;
	livetraining = msg.value.asBoolean;
	livetraining.postln;
};

Button(win,Rect(1270,280,100,20))
.states_([["Show Data"],["Hide Data"]])
.action_{
	arg msg;
	visibility = msg.value.asBoolean;
	visibility.postln;
};

Button(win,Rect(1380,10,100,100))
.states_([
        ["Start VTrack", Color.white, Color.green(0.6)],
        ["Stop VTrack",  Color.white, Color.red(0.6)],
    ])
.action_{
	arg msg;

	if (msg.value == 1) {
		~pid = "bash -c 'conda run -n mediapipe python /home/pemb/python/mediapipe/bodyPose.py'".unixCmd;
		
        } {
            "pkill -f bodyPose.py".unixCmd;  // kill it on toggle off
        };
    });
	
};
	
//s.sync;


~synthbus = Bus.audio(s, 2);
//NN.load(\ravePerc, "/home/pemb/Music/Supercollider/nnRave/models/voice-multi-b2048-r48000-z11.ts");

/*
synth = {

	arg predicting = 0;

	var val, mp, trig;

	//comvert the mpbuf to control 
	mp = FluidBufToKr.kr(mpbuf);
	//trigger if the control changes
	trig = Mix(Changed.kr(mp));
	//The trigger is activated when the argument
	//predicting is 1. Otherwise the trigger will
	//be always 0, therefore not triggering.
	//The mlp claculates the outpus of the nn
	//only when is triggered.
	mlp.kr(trig * predicting, mpbuf, paramsbuf);
	val = FluidBufToKr.kr(paramsbuf);
	//Define a OSC message to be send with the
	//values of the paramsbuffer to the address
	// /paramsChnaged, everytome the values in
	//paramsbuf change. This is values from the
	//sound server to the language interpreter
	SendReply.kr(Mix(Changed.kr(val)), "/paramsChanged", val);


	

}
	*/
//SYNTH
synth = {

	arg predicting = 0;
	
	//define that the dataset control the arguments of the synth.
	//var val = FluidBufToKr.kr(paramsbuf); 

	var in, latent, modLatent, prior, rlmod = 20;

	var osc1, osc2, feed1, feed2, base1=69, base2=69, base3 = 130, loud1, loud2, sig;

	var val, mp, trig;

	//comvert the mpbuf to control 
	mp = FluidBufToKr.kr(mpbuf);
	//trigger if the control changes
	trig = Mix(Changed.kr(mp));
	//The trigger is activated when the argument
	//predicting is 1. Otherwise the trigger will
	//be always 0, therefore not triggering.
	//The mlp claculates the outpus of the nn
	//only when is triggered.
	mlp.kr(trig * predicting, mpbuf, paramsbuf);
	val = FluidBufToKr.kr(paramsbuf);
	//Define a OSC message to be send with the
	//values of the paramsbuffer to the address
	// /paramsChnaged, everytome the values in
	//paramsbuf change. This is values from the
	//sound server to the language interpreter
	SendReply.kr(Mix(Changed.kr(val)), "/paramsChanged", val);
	
	
	#feed2, feed1 = LocalIn.ar(2);

	//only for testing is we get the data from MP. 
	//	FluidBufToKr.kr(~mpbuf).poll;

	loud1 = FluidLoudness.kr(feed1,[\loudness],1,0,hopSize:64);
	loud2 = FluidLoudness.kr(feed2,[\loudness],1,0,hopSize:64);

	osc1 = SinOsc.ar((((feed1 * val[0]) +  val[1]) * base1).midicps,mul:(val[2] * 50).dbamp).atan;
	osc1 = MoogFF.ar(osc1,(base3-(val[3]*(loud2.clip(-120,0)+120))).lag(128/44100).midicps,val[4]*3.5);

	osc2 = SinOsc.ar((((feed2 * val[5]) +  val[6]) * base2).midicps,mul: (val[7] * 50).dbamp).atan;
	osc2 = MoogFF.ar(osc2,(base3-(val[8]*(loud1.clip(-120,0)+120))).lag(128/44100).midicps,val[9]*3.5);

	//		Out.ar(~synthbus,LeakDC.ar([osc1,osc2],mul:0.1));

	LocalOut.ar([osc1,osc2]);
	
sig = [osc1,osc2] * 0.1;

	/*
    // encode sound input to RAVE latent space
    latent = NN(\ravePerc, \encode).ar(sig);
    // add a random modulation to every dimension
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(val[10] * rlmod).range(-0.5, 0.5)
    };
    // resynthesize modulated latents
	//    NN(\ravePerc, \decode).ar(modLatent);

	
	FreeVerb.ar(NN(\ravePerc, \decode).ar(modLatent), LFNoise2.kr(1).range(0,1), LFNoise2.kr(1).range(0,1), LFNoise2.kr(1).range(0,1));
	*/
}.play;


//OSCdef to receive the masse from the audio server
//to the language interpreter
OSCdef(\paramsChanged,{
	arg msg;
	defer{
		mss.value_(msg[3..]);
	};
},"/paramsChanged");


)

s.meter;
s.plotTree;



