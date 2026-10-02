const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../autopcr/http_server/ClientApp/mastery.html'), 'utf8');
const elements = new Map();
const cells = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {checked: false, writes: 0,
    set innerHTML(value) {this.writes++;this.html=value;},
    addEventListener() {},
    querySelector(selector) {
      const key=selector.match(/data-key="([^"]+)"/)[1];
      if (!cells.has(key)) cells.set(key, {parts:new Map(),classList:{toggle() {}},
        querySelector(part) {if (!this.parts.has(part)) this.parts.set(part, {});return this.parts.get(part);}});
      return selector.includes('.cost-hint')?cells.get(key).querySelector('.cost-hint'):cells.get(key);
    }});
  return elements.get(id);
}};
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(
  '    loadAccounts();',
  `globalThis.engine = {takeCategory, setSteps, stepsFor, stateAt, resetSimulations, renderSimulation,
    configure(payload) { data=payload; meta=payload.metadata; initialize(); },
    stocks() { return {...stock}; }};`
);
const context = vm.createContext({document});
vm.runInContext(source, context);
const engine = context.engine;
const normalize = value => JSON.parse(JSON.stringify(value));

// Original per-fragment algorithm, retained as an independent equivalence oracle.
function referenceTake(stock, itemAt, level, need, universal) {
  const consumed = new Map();
  function acquire(work, lv) {
    if (lv > 1) {
      const trial = {...work};
      const parts = [];
      let success = true;
      for (let i = 0; i < 3; i++) {
        const part = acquire(trial, lv - 1);
        if (!part) {success = false; break;}
        parts.push(...part);
      }
      if (success) {Object.assign(work, trial); return parts;}
    }
    const item = String(itemAt(lv) || '');
    if (!item || !(work[item] > 0)) return null;
    work[item]--;
    return [{item, converted: lv < level, universal}];
  }
  let supplied = 0;
  while (supplied < need) {
    const parts = acquire(stock, level);
    if (!parts) break;
    for (const part of parts) {
      const key = JSON.stringify(part);
      const previous = consumed.get(key);
      if (previous) previous.count++;
      else consumed.set(key, {...part, count: 1});
    }
    supplied++;
  }
  return {takes: [...consumed.values()], supplied};
}
let seed = 20261002;
function random(limit) {seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%limit;}
function sortPlan(plan) {return {...normalize(plan), takes: normalize(plan.takes).sort((a,b)=>a.item.localeCompare(b.item))};}
for (let i=0;i<1000;i++) {
  const initial={};
  for (let lv=1;lv<=5;lv++) initial[String(80000+lv)]=random(180);
  const actual={...initial},expected={...initial};
  const level=1+random(5),need=random(100),universal=!!random(2);
  const itemAt=lv=>80000+lv;
  assert.deepEqual(sortPlan(engine.takeCategory(actual,itemAt,level,need,universal)),
    sortPlan(referenceTake(expected,itemAt,level,need,universal)));
  assert.deepEqual(actual,expected);
}

const payload={metadata:{roles:{1:'攻击型'},mastery:{1:{1:{items:{1:81101,2:81102,3:81103,4:81104,5:81105}}}},
  universal:{1:80001,2:80002,3:80003,4:80004,5:80005},costs:[20,40,50,60,70,90]},
  unit_role_list:[{unit_role_id:1,slot_level_1:1,enhance_level_1:0}],stocks:{81101:39,80001:1}};
engine.configure(payload);
assert.equal(engine.setSteps('1-1',1,false),0); // Insufficient plans do not consume stock.
assert.deepEqual(normalize(engine.stocks()),payload.stocks);
assert.equal(engine.setSteps('1-1',1,true),1);
assert.deepEqual(normalize(engine.stateAt('1-1')),{level:1,stars:1});
assert.equal(engine.setSteps('1-1',0,true),0);
assert.deepEqual(normalize(engine.stocks()),payload.stocks); // Refund includes universal fragments.

payload.unit_role_list[0].enhance_level_1=5;
payload.stocks={81101:6,81102:18}; // 6 Lv1 -> 2 Lv2 plus 18 raw Lv2 pays the rank-up cost.
engine.configure(payload);
assert.equal(engine.setSteps('1-1',1,false),1);
assert.deepEqual(normalize(engine.stateAt('1-1')),{level:2,stars:0});
engine.resetSimulations();
assert.deepEqual(normalize(engine.stocks()),payload.stocks);
engine.renderSimulation(['1-1']);
assert.equal(elements.get('roles').writes,0);
assert.equal(elements.get('inventory').writes,0);
assert.equal(elements.get('costs')?.writes||0,0);
const retainedCell=cells.get('1-1');
engine.renderSimulation(['1-1']);
assert.equal(cells.get('1-1'),retainedCell);
console.log('Passed 1,000 conversion equivalence cases, atomic spend, universal refund, level transition and incremental rendering checks.');

const benchmarkStock={};
for(let i=0;i<165;i++) benchmarkStock[String(80001+i)]=100000;
const now=()=>process.hrtime.bigint();
const startOld=now();
for(let i=0;i<20;i++) referenceTake({...benchmarkStock},lv=>80000+lv,5,90,false);
const oldMs=Number(now()-startOld)/1e6;
const startNew=now();
for(let i=0;i<20;i++) engine.takeCategory({...benchmarkStock},lv=>80000+lv,5,90,false);
const newMs=Number(now()-startNew)/1e6;
console.log(`Batch conversion benchmark (20 plans): old ${oldMs.toFixed(1)}ms, new ${newMs.toFixed(1)}ms.`);
