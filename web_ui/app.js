const API_BASE = "http://127.0.0.1:8000";
document.getElementById("api-url").textContent = API_BASE;

//Elements
const qEl = document.getElementById("q");
const goBtn = document.getElementById("go");
const clearBtn = document.getElementById("clear");
const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");
const beforeEl = document.getElementById("before");
const rootEl = document.getElementById("root");
const folderStatusEl = document.getElementById("folder-status");
const idxBm25Btn = document.getElementById("index-bm25");
const idxChromaBtn = document.getElementById("index-chroma");
const idxBothBtn = document.getElementById("index-both");
const browseBtn = document.getElementById("browse-root");

//Engine & chunk toggles
let engine = "router", k = 6;

function makeToggles(id, cb){
  document.querySelectorAll(`#${id} button`).forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(`#${id} button`).forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      cb(btn.dataset.val);
    });
  });
}
makeToggles("engine-toggles", v=>engine=v);
makeToggles("chunk-toggles", v=>k=Number(v));

//Load current workspace (if exists)
async function loadCurrentWorkspace(){
  try {
    const res = await fetch(`${API_BASE}/list-workspaces`);
    if (!res.ok) return;
    const data = await res.json();
    if (data?.length) {
      const current = data[0];
      rootEl.value = current;
      folderStatusEl.textContent = `Currently indexed: ${current}`;
    } else {
      folderStatusEl.textContent = "Currently indexed: none";
    }
  } catch (e) {
    console.warn("Workspace check failed:", e);
  }
}
loadCurrentWorkspace();

//Helpers
function showStatus(text,isError=false){
  statusEl.textContent=text;
  statusEl.className=isError?"error mini":"muted mini";
}
function clearOutput(){outEl.innerHTML="";showStatus("");}
clearBtn.onclick=()=>{qEl.value="";clearOutput();};

//API calls
async function callAPI(url,body){
  const res=await fetch(API_BASE+url,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
  if(!res.ok)throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
async function callRouter(q){return callAPI("/route",{question:q,k,before:beforeEl.value||null});}
async function callAgent(q){return callAPI("/agent/route",{question:q,k,before:beforeEl.value||null,steps:4});}

async function pickFolder(){
  try{
    const res=await fetch(`${API_BASE}/pick-folder`);
    const data=await res.json();
    if(data.root){rootEl.value=data.root;folderStatusEl.textContent=`Currently indexed: ${data.root}`;}
  }catch(e){showStatus("Folder picker failed",true);}
}

async function callIndex(kind,root){
  const ep=kind==="bm25"?"/index/bm25":kind==="chroma"?"/index/chroma":null;
  if(!ep)throw new Error("Invalid kind");
  return callAPI(ep,{root});
}

async function runIndex(kind){
  const root=rootEl.value.trim();
  if(!root)return showStatus("Enter folder path first",true);
  const btn=kind==="bm25"?idxBm25Btn:kind==="chroma"?idxChromaBtn:idxBothBtn;
  btn.disabled=true;btn.innerHTML="<span class='spinner'></span>Indexing…";
  try{
    if(kind==="both"){await callIndex("bm25",root);await callIndex("chroma",root);}
    else await callIndex(kind,root);
    folderStatusEl.textContent=`Currently indexed: ${root}`;
    showStatus("Indexing complete");
  }catch(e){showStatus(e.message,true);}
  finally{btn.disabled=false;btn.textContent=btn.id.includes("bm25")?"Index BM25":btn.id.includes("chroma")?"Index Chroma":"Index Both";}
}

//Search
async function run(){
  const q=qEl.value.trim();
  if(!q)return showStatus("Please enter a question",true);
  goBtn.disabled=true;goBtn.innerHTML="<span class='spinner'></span>Searching…";
  try{
    const resp=engine==="agent"?await callAgent(q):await callRouter(q);
    render(resp);
  }catch(e){showStatus(e.message,true);}
  finally{goBtn.disabled=false;goBtn.textContent="Search";}
}

//Render
function render(resp){
  clearOutput();
  if(resp.mode==="files")renderFiles(resp);
  else if(resp.mode==="answer")renderAnswer(resp);
}
function el(tag,cls,txt){const n=document.createElement(tag);if(cls)n.className=cls;if(txt)n.textContent=txt;return n;}
function renderFiles(r){
  if(!r.files?.length)return outEl.appendChild(el("div","muted","No files matched."));
  r.files.forEach((f,i)=>{
    const card=el("div","card");
    card.appendChild(el("h4",null,`${i+1}. ${f.path}`));
    card.appendChild(el("div","muted",f.preview||""));
    outEl.appendChild(card);
  });
}
function renderAnswer(r) {
  const card = el("div", "card");
  card.appendChild(el("h4", null, "Answer"));

  //Format main answer text
  let clean = (r.answer || "")
    .replace(/^=+|=+$/gm, "")
    .replace(/^ANSWER[:\- ]*/i, "")
    .replace(/^SOURCES[:\- ]*/i, "")
    .replace(/\n{2,}/g, "\n")
    .trim();

  const html = clean
    .split("\n")
    .map(line => `<p>${line}</p>`)
    .join("");

  const ansDiv = document.createElement("div");
  ansDiv.className = "answer";
  ansDiv.innerHTML = html;
  card.appendChild(ansDiv);

  //Citations section
  if (r.citations?.length) {
    const cwrap = el("div", "chips");
    card.appendChild(el("div", "muted", "Citations:"));

    r.citations.forEach((c, i) => {
      const chip = document.createElement("span");
      chip.className = "chip clickable";
      chip.textContent = `[${i + 1}] ${c.path} · p.${c.page}`;

      //Click handler to open file via backend endpoint
      chip.addEventListener("click", async () => {
        try {
          await fetch(`${API_BASE}/open-file`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: c.path })
          });
        } catch (err) {
          alert("Failed to open file: " + err.message);
        }
      });

      cwrap.appendChild(chip);
    });

    card.appendChild(cwrap);
  }

  outEl.appendChild(card);
}

//Event bindings
goBtn.onclick=run;
qEl.addEventListener("keydown",e=>{if(e.ctrlKey&&e.key==="Enter")run();});
browseBtn.onclick=pickFolder;
idxBm25Btn.onclick=()=>runIndex("bm25");
idxChromaBtn.onclick=()=>runIndex("chroma");
idxBothBtn.onclick=()=>runIndex("both");
