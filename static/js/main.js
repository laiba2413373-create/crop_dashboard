/*=========================================
UPLOAD TOGGLE
=========================================*/

function toggleUpload(show){

    const upload=document.getElementById("uploadDiv");

    if(upload){

        upload.style.display=show?"block":"none";

    }

}

/*=========================================
PAGE LOADER
=========================================*/

document.addEventListener("DOMContentLoaded",()=>{

document.body.classList.add("fade-up");

});

/*=========================================
BUTTON LOADING
=========================================*/

const form=document.querySelector("form");

if(form){

form.addEventListener("submit",function(){

const btn=this.querySelector(".btn-ai");

if(btn){

btn.disabled=true;

btn.innerHTML=`
<span class="spinner-border spinner-border-sm me-2"></span>
Running AI Model...
`;

}

});

}

/*=========================================
COUNT UP ANIMATION
=========================================*/

function animateValue(el){

const target=parseFloat(el.innerText);

if(isNaN(target)) return;

let count=0;

const speed=target/80;

const interval=setInterval(()=>{

count+=speed;

if(count>=target){

count=target;

clearInterval(interval);

}

el.innerHTML=count.toFixed(2)+"%";

},20);

}

document.addEventListener("DOMContentLoaded",()=>{

document.querySelectorAll(".accuracy-value").forEach(e=>{

animateValue(e);

});

});

/*=========================================
CARD HOVER EFFECT
=========================================*/

document.querySelectorAll(".glass-card").forEach(card=>{

card.addEventListener("mousemove",(e)=>{

const rect=card.getBoundingClientRect();

const x=e.clientX-rect.left;

const y=e.clientY-rect.top;

card.style.background=

`radial-gradient(circle at ${x}px ${y}px,
rgba(255,255,255,.12),
rgba(255,255,255,.05) 60%)`;

});

card.addEventListener("mouseleave",()=>{

card.style.background="rgba(255,255,255,.08)";

});

});

/*=========================================
SCROLL ANIMATION
=========================================*/

const observer=new IntersectionObserver(entries=>{

entries.forEach(entry=>{

if(entry.isIntersecting){

entry.target.style.opacity="1";

entry.target.style.transform="translateY(0)";

}

});

});

document.querySelectorAll(".glass-card").forEach(card=>{

card.style.opacity="0";

card.style.transform="translateY(40px)";

card.style.transition=".8s";

observer.observe(card);

});

/*=========================================
BACK TO TOP
=========================================*/

const topBtn=document.createElement("button");

topBtn.innerHTML="↑";

topBtn.id="topBtn";

document.body.appendChild(topBtn);

topBtn.style.position="fixed";
topBtn.style.right="25px";
topBtn.style.bottom="25px";
topBtn.style.width="55px";
topBtn.style.height="55px";
topBtn.style.borderRadius="50%";
topBtn.style.border="none";
topBtn.style.cursor="pointer";
topBtn.style.fontSize="22px";
topBtn.style.display="none";
topBtn.style.zIndex="999";
topBtn.style.background="linear-gradient(135deg,#00f5c3,#37b7ff)";
topBtn.style.color="#07131d";
topBtn.style.fontWeight="700";
topBtn.style.boxShadow="0 0 25px rgba(0,245,195,.35)";

window.addEventListener("scroll",()=>{

if(window.scrollY>300){

topBtn.style.display="block";

}else{

topBtn.style.display="none";

}

});

topBtn.onclick=()=>{

window.scrollTo({

top:0,

behavior:"smooth"

});

};

/*=========================================
TILT EFFECT
=========================================*/

document.querySelectorAll(".feature-card").forEach(card=>{

card.addEventListener("mousemove",(e)=>{

const rect=card.getBoundingClientRect();

const x=e.clientX-rect.left;

const y=e.clientY-rect.top;

const rotateX=((y-rect.height/2)/15);

const rotateY=((rect.width/2-x)/15);

card.style.transform=

`perspective(1000px)
rotateX(${rotateX}deg)
rotateY(${rotateY}deg)
translateY(-8px)`;

});

card.addEventListener("mouseleave",()=>{

card.style.transform="rotateX(0) rotateY(0)";

});

});