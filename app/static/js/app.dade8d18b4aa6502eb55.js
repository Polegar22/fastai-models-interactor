webpackJsonp([0],[,,function(t,e,n){"use strict";var a=n(11),s=n.n(a),i=window.location;s.a.defaults.baseURL=i.protocol+"//"+i.hostname+":"+i.port+"/api",e.a=s.a},,,,,function(t,e,n){"use strict";var a=n(30);e.a={name:"App",components:{Menu:a.a},data:function(){return{openDrawer:!1}}}},function(t,e,n){"use strict";e.a={data:function(){return{items:[{title:"Home",path:"/",icon:"home"},{title:"CNN Image Recognition",path:"image-recognition",icon:"image_search"},{title:"CNN Sound Detection",path:"sound-detection",icon:"hearing"},{title:"Natural Language Processing",path:"nlp",icon:"notes"},{title:"Unet Image reconstruction",path:"image-reconstruction",icon:"picture_in_picture"}]}},methods:{goTo:function(t){this.$router.push({path:t})}}}},function(t,e){},function(t,e,n){"use strict";var a=n(41),s=n.n(a),i=n(42),r=(n.n(i),n(2));e.a={name:"UploadImage",components:{VueDropzone:s.a},data:function(){return{dropzoneOptions:{maxFiles:1,thumbnailHeight:250,maxFilesize:5,addRemoveLinks:!0,headers:{"Cache-Control":null,"X-Requested-With":null}},aiResponse:"Does the AI things you are beautiful ?"}},created:function(){this.dropzoneOptions.url=r.a.defaults.baseURL+"/image-recognition"},methods:{uploadSucceded:function(t,e){this.aiResponse=e.result},fileDeleted:function(){this.aiResponse="Does the AI things your beautiful ?"}}}},,,,,,,function(t,e,n){"use strict";var a=n(2);e.a={name:"UploadSound",data:function(){return{aiResponse:"Upload a sound.",url:"",headers:{"Cache-Control":null,"X-Requested-With":null}}},created:function(){this.url=a.a.defaults.baseURL+"/sound-detection"},methods:{uploadSucceded:function(t){this.aiResponse=t.data.result}}}},function(t,e,n){"use strict";var a=n(2),s=n(19);e.a={components:{CustomTitle:s.a},data:function(){return{danaherText:"",entryText:"",nbWords:45,randomness:50,sending:!1}},computed:{percentRdm:function(){return this.randomness/100}},methods:{sendEntryText:function(){var t=this;this.sending=!0,this.danaherText="";var e=new FormData;e.append("entry_text",this.entryText),e.append("nb_words",this.nbWords),e.append("randomness",this.percentRdm),a.a.post("/nlp",e).then(function(e){t.danaherText=e.data.result,t.sending=!1})}}}},function(t,e,n){"use strict";var a=n(20),s=n(68);var i=function(t){n(67)},r=n(1)(a.a,s.a,!1,i,"data-v-38a536af",null);e.a=r.exports},function(t,e,n){"use strict";e.a={name:"UploadSound",props:{title:{type:String,default:""},subtitle:{type:String,default:""}}}},function(t,e,n){"use strict";var a=n(2),s=n(72),i=n(19);e.a={components:{DrawableCanvas:s.a,CustomTitle:i.a},data:function(){return{sending:!1,error:""}},methods:{sendCrapifiedImage:function(){var t=this;this.sending=!0;var e=this.$refs.drawableCanvas.getCrapifiedImage(),n=new FormData;n.append("image",e),a.a.post("/image-reconstruction",n).then(function(e){t.sending=!1,t.$refs.drawableCanvas.setImageToCanvas(e.data)}).catch(function(e){t.sending=!1,t.error=e.response.status+" "+e.response.statusText})}}}},function(t,e,n){"use strict";e.a={data:function(){return{canvas:null,ctx:null,clientWidth:null,maxWidth:null,rect:{},drag:!1,canDraw:!1}},mounted:function(){this.canvas=this.$refs.drawableCanvas,this.ctx=this.canvas.getContext("2d"),this.clientWidth=document.documentElement.clientWidth-20,this.maxWidth=this.clientWidth<600?this.clientWidth:600,this.canvas.addEventListener("mousedown",this.mouseDown,!1),this.canvas.addEventListener("mouseup",this.mouseUp,!1),this.canvas.addEventListener("mousemove",this.mouseMove,!1),this.canvas.addEventListener("touchstart",this.mouseDown,!1),this.canvas.addEventListener("touchend",this.mouseUp,!1),this.canvas.addEventListener("touchmove",this.mouseMove,!1)},methods:{openDialog:function(){document.getElementById("input").click()},fileSelected:function(t){var e=new FileReader,n=this;this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height),e.onload=function(t){var e=new Image;e.onload=function(){var t=e.width,a=e.height;e.width>n.maxWidth&&(t=n.maxWidth,a=e.height*(t/e.width)),n.canvas.width=t,n.canvas.height=a,n.ctx.drawImage(e,0,0,e.width,e.height,0,0,n.canvas.width,n.canvas.height),n.canDraw=!0},e.src=t.target.result},e.readAsDataURL(t[0])},mouseDown:function(t){var e=t.pageX?t.pageX:t.changedTouches[0].pageX,n=t.pageY?t.pageY:t.changedTouches[0].pageY;this.rect.startX=e-this.canvas.offsetLeft,this.rect.startY=n-this.canvas.offsetTop,this.drag=!0},mouseUp:function(){this.drag=!1,this.canDraw&&(this.$emit("drawEnded"),this.canDraw=!1)},mouseMove:function(t){if(this.drag&&this.canDraw){var e=t.pageX?t.pageX:t.changedTouches[0].pageX,n=t.pageY?t.pageY:t.changedTouches[0].pageY,a=this.canvas.width/this.canvas.clientWidth,s=this.canvas.height/this.canvas.clientHeight;this.rect.w=e-this.canvas.offsetLeft-this.rect.startX,this.rect.h=n-this.canvas.offsetTop-this.rect.startY,this.ctx.fillRect(this.rect.startX*a,this.rect.startY*s,this.rect.w*a,this.rect.h*s)}},getCrapifiedImage:function(){return this.canvas.toDataURL()},setImageToCanvas:function(t){var e=new Image,n=this;e.onload=function(){n.canvas.width=n.maxWidth,n.canvas.height=e.height*(n.maxWidth/e.width),n.ctx.clearRect(0,0,n.canvas.width,n.canvas.height),n.ctx.drawImage(e,0,0,e.width,e.height,0,0,n.canvas.width,n.canvas.height),n.canDraw=!0},e.src="data:image/jpeg;base64,"+t}}}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var a=n(24),s=(n.n(a),n(3)),i=n(28),r=n(34),o=n(76),c=n.n(o),u=n(77),d=n.n(u),l=n(78),h=n(11),v=n.n(h);s.default.use(c.a,{theme:l.a}),s.default.use(d.a),s.default.prototype.$http=v.a,s.default.config.productionTip=!1,new s.default({el:"#app",router:r.a,components:{App:i.a},template:"<App/>"})},function(t,e){},,,,function(t,e,n){"use strict";var a=n(7),s=n(33);var i=function(t){n(29)},r=n(1)(a.a,s.a,!1,i,null,null);e.a=r.exports},function(t,e){},function(t,e,n){"use strict";var a=n(8),s=n(32);var i=function(t){n(31)},r=n(1)(a.a,s.a,!1,i,"data-v-d232e3cc",null);e.a=r.exports},function(t,e){},function(t,e,n){"use strict";var a={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("v-list",{staticClass:"pt-0",attrs:{dense:""}},t._l(t.items,function(e){return n("v-list-tile",{key:e.title,on:{click:function(n){return t.goTo(e.path)}}},[n("v-list-tile-action",[n("v-icon",{attrs:{large:""}},[t._v("\n        "+t._s(e.icon)+"\n      ")])],1),t._v(" "),n("v-list-tile-content",[n("v-list-tile-title",[t._v("\n        "+t._s(e.title)+"\n      ")])],1)],1)}),1)},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("v-app",[n("div",{attrs:{id:"app"}},[n("header",[n("v-btn",{staticClass:"btn-drawer",attrs:{flat:"",dark:"",icon:""},on:{click:function(e){t.openDrawer=!0}}},[n("v-icon",[t._v("menu")])],1),t._v(" "),n("v-navigation-drawer",{staticClass:"drawer",attrs:{absolute:"",temporary:""},model:{value:t.openDrawer,callback:function(e){t.openDrawer=e},expression:"openDrawer"}},[n("Menu")],1)],1),t._v(" "),n("main",[n("router-view")],1)])])},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a=n(3),s=n(35),i=n(36),r=n(39),o=n(62),c=n(65),u=n(70);a.default.use(s.a),e.a=new s.a({mode:"history",routes:[{path:"/",name:"Home",component:i.default},{path:"/image-recognition",name:"ImageRecognition",component:r.a},{path:"/sound-detection",name:"SoundDetection",component:o.a},{path:"/nlp",name:"NaturalLanguageProcessing",component:c.a},{path:"/image-reconstruction",name:"ImageReconstruction",component:u.a}]})},,function(t,e,n){"use strict";var a=n(9),s=n.n(a),i=n(38);var r=function(t){n(37)},o=n(1)(s.a,i.a,!1,r,"data-v-d43b305a",null);e.default=o.exports},function(t,e){},function(t,e,n){"use strict";var a={render:function(){this.$createElement;this._self._c;return this._m(0)},staticRenderFns:[function(){var t=this.$createElement,e=this._self._c||t;return e("div",[e("h1",[this._v("Select in the menu the model you want to interract with.")])])}]};e.a=a},function(t,e,n){"use strict";var a=n(10),s=n(61);var i=function(t){n(40)},r=n(1)(a.a,s.a,!1,i,"data-v-fb10e73a",null);e.a=r.exports},function(t,e){},,function(t,e){},,,,,,,,,,,,,,,,,,,function(t,e,n){"use strict";var a={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"imageUpload"},[n("h2",[t._v(t._s(t.aiResponse))]),t._v(" "),n("vue-dropzone",{ref:"myVueDropzone",staticStyle:{"margin-top":"30px"},attrs:{id:"dropzone",options:t.dropzoneOptions,"duplicate-check":"","use-custom-slot":!0},on:{"vdropzone-success":t.uploadSucceded,"vdropzone-removed-file":t.fileDeleted}},[n("div",[n("h3",{staticClass:"dropzone-custom-title"},[t._v("\n        Drag and drop to upload content!\n      ")]),t._v(" "),n("div",{staticClass:"subtitle"},[t._v("\n        ...or click to select a file from your computer\n      ")])])])],1)},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a=n(17),s=n(64);var i=function(t){n(63)},r=n(1)(a.a,s.a,!1,i,null,null);e.a=r.exports},function(t,e){},function(t,e,n){"use strict";var a={render:function(){var t=this.$createElement,e=this._self._c||t;return e("div",{staticClass:"audioUpload"},[e("h2",[this._v(this._s(this.aiResponse))]),this._v(" "),e("audio-recorder",{staticStyle:{"margin-top":"30px"},attrs:{"upload-url":this.url,attempts:1,time:.08,"show-download-button":!1,headers:this.headers,"successful-upload":this.uploadSucceded}})],1)},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a=n(18),s=n(69);var i=function(t){n(66)},r=n(1)(a.a,s.a,!1,i,"data-v-1351434e",null);e.a=r.exports},function(t,e){},function(t,e){},function(t,e,n){"use strict";var a={render:function(){var t=this.$createElement,e=this._self._c||t;return e("div",[e("h2",[this._v(this._s(this.title))]),this._v(" "),e("h3",{staticClass:"subtitle"},[this._v("\n    "+this._s(this.subtitle)+"\n  ")])])},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"nlpUpload"},[n("custom-title",{attrs:{title:"John Danaher Generator",subtitle:"AI will complete your sentence in the style of John Danaher"}}),t._v(" "),n("div",[n("div",{staticClass:"nlpInputs"},[n("v-text-field",{attrs:{label:"Entry Text"},model:{value:t.entryText,callback:function(e){t.entryText=e},expression:"entryText"}}),t._v(" "),n("v-text-field",{attrs:{label:"Number of words"},model:{value:t.nbWords,callback:function(e){t.nbWords=e},expression:"nbWords"}}),t._v(" "),n("v-slider",{attrs:{label:"Randomness","thumb-label":"always"},model:{value:t.randomness,callback:function(e){t.randomness=e},expression:"randomness"}})],1),t._v(" "),n("v-btn",{directives:[{name:"show",rawName:"v-show",value:t.entryText&&!t.sending,expression:"entryText && !sending"}],attrs:{color:"accent"},on:{click:t.sendEntryText}},[t._v("\n      Send\n    ")]),t._v(" "),n("v-progress-circular",{directives:[{name:"show",rawName:"v-show",value:t.sending,expression:"sending"}],attrs:{indeterminate:"",color:"accent"}})],1),t._v(" "),t.danaherText?n("span",{staticClass:"danaherText"},[t._v("\n    "+t._s(t.danaherText)+"\n  ")]):t._e()],1)},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a=n(21),s=n(75);var i=function(t){n(71)},r=n(1)(a.a,s.a,!1,i,"data-v-24cc0ce5",null);e.a=r.exports},function(t,e){},function(t,e,n){"use strict";var a=n(22),s=n(74);var i=function(t){n(73)},r=n(1)(a.a,s.a,!1,i,"data-v-094e9ad8",null);e.a=r.exports},function(t,e){},function(t,e,n){"use strict";var a={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"drawableContainer"},[n("div",{staticStyle:{width:"300px"}},[n("v-text-field",{attrs:{label:"Upload a file","prepend-icon":"attach_file"},nativeOn:{click:function(e){return t.openDialog(e)}}}),t._v(" "),n("input",{directives:[{name:"show",rawName:"v-show",value:!1,expression:"false"}],attrs:{id:"input",type:"file"},on:{change:function(e){return t.fileSelected(e.target.files)}}})],1),t._v(" "),n("canvas",{ref:"drawableCanvas"})])},staticRenderFns:[]};e.a=a},function(t,e,n){"use strict";var a={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"imageReconstruction"},[n("custom-title",{attrs:{title:"Let the IA correct your imperfections : ",subtitle:"Upload a face and select the parts you want to clean"}}),t._v(" "),n("drawable-canvas",{ref:"drawableCanvas",staticClass:"marginTop",on:{drawEnded:t.sendCrapifiedImage}}),t._v(" "),n("div",{staticClass:"marginTop"},[n("v-progress-circular",{directives:[{name:"show",rawName:"v-show",value:t.sending,expression:"sending"}],attrs:{indeterminate:"",color:"accent"}})],1),t._v(" "),t.error?n("span",[t._v(t._s(t.error))]):t._e()],1)},staticRenderFns:[]};e.a=a},,,function(t,e,n){"use strict";e.a={primary:"#2176ae",secondary:"#83d4e6",accent:"#E1616C",success:"#3fc380"}}],[23]);