/* Records a narrated screencast of the whole Clutch IQ UI.
 *
 *   uvicorn backend_app:app --port 8000     # in another shell
 *   node tools/record_walkthrough.js [chapter...]   # e.g. "node tools/record_walkthrough.js 1 2"
 *
 * Narration is burned in as captions (no TTS in this environment). Fonts are
 * served from a local @fontsource copy when VW_FONTS points at one, so the
 * recording shows the same typography as a networked browser.
 */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const URL = process.env.VW_URL || 'http://localhost:8000/';
const OUT = process.env.VW_OUT || '/tmp/vw';
const FONTS = process.env.VW_FONTS || '';
const DATA = process.env.VW_DATA || path.resolve(__dirname, '..', 'sample_datasets');
const ONLY = process.argv.slice(2).map(Number).filter(n => !isNaN(n));
const SPEED = Number(process.env.VW_SPEED || 1);

const CPS = 15.5;                      // reading speed when there is no voice track
const DUMP = !!process.env.VW_DUMP;    // print the narration script and exit (no browser)
const AUDIO = process.env.VW_AUDIO || '';  // directory holding narration.json + wavs
const crypto = require('crypto');
const NARR = AUDIO && fs.existsSync(path.join(AUDIO, 'narration.json'))
  ? JSON.parse(fs.readFileSync(path.join(AUDIO, 'narration.json'), 'utf8')) : null;
const CAPS = [];                       // every caption, in order (dump + sync log)
const speechKey = t => crypto.createHash('sha1')
  .update(String(t).replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim()).digest('hex').slice(0, 16);
const missing = [];
// how long a caption stays up: the spoken line plus a breath, or reading speed
function capDur(text) {
  const k = speechKey(text);
  if (NARR && NARR[k]) return NARR[k].dur * 1000 + 420;
  if (NARR) missing.push(text.slice(0, 50));
  return Math.min(9500, Math.max(2500, (text.length / CPS) * 1000)) * SPEED;
}
const dur = t => capDur(t);

const CSS = `
#vwCap{position:fixed;left:0;right:0;bottom:0;z-index:99999;padding:20px 48px 20px;
  background:linear-gradient(180deg,rgba(5,9,14,0) 0%,rgba(5,9,14,.92) 34%,rgba(5,9,14,.985) 100%);
  font:500 20.5px/1.45 "IBM Plex Sans",system-ui,sans-serif;color:#eaf2fb;letter-spacing:.1px;
  display:block;min-height:64px;opacity:0;transition:opacity .25s}
#vwCap>span{display:block;max-width:1380px}
#vwCap.on{opacity:1}
#vwCap b{color:#39d6f2;font-weight:600}
#vwCap i{color:#f06fd0;font-style:normal}
#vwChip{position:fixed;right:18px;bottom:78px;z-index:99999;padding:5px 12px;border-radius:999px;
  background:rgba(10,165,196,.16);border:1px solid rgba(57,214,242,.5);color:#8fe6f7;
  font:600 12.5px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;text-transform:uppercase;opacity:0;transition:opacity .3s}
#vwChip.on{opacity:1}
#vwCard{position:fixed;inset:0;z-index:99998;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:18px;background:rgba(4,7,11,.93);opacity:0;pointer-events:none;transition:opacity .45s}
#vwCard.on{opacity:1}
#vwCard h1{font:600 54px/1.1 "Chakra Petch",system-ui,sans-serif;color:#eaf2fb;margin:0;text-align:center;letter-spacing:.5px}
#vwCard h1 span{color:#39d6f2}
#vwCard p{font:400 22px/1.5 "IBM Plex Sans",system-ui,sans-serif;color:#9fb3c8;margin:0;max-width:60ch;text-align:center}
#vwCard ul{font:400 19px/1.75 "IBM Plex Sans",system-ui,sans-serif;color:#c6d4e4;margin:0;max-width:66ch}
#vwCur{position:fixed;z-index:99997;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
  border:2px solid #39d6f2;background:rgba(57,214,242,.22);box-shadow:0 0 14px rgba(57,214,242,.55);
  pointer-events:none;opacity:0;transition:opacity .2s,left .42s cubic-bezier(.3,.9,.3,1),top .42s cubic-bezier(.3,.9,.3,1)}
#vwCur.on{opacity:1}
#vwCur.tap{transform:scale(.6);transition:transform .12s}
.vw-ring{outline:2px solid #39d6f2 !important;outline-offset:3px;border-radius:6px;
  box-shadow:0 0 0 6px rgba(57,214,242,.16) !important}
body{cursor:none !important}
/* sync strip: flashes when a caption starts; cropped off the delivered video */
#vwMark{position:fixed;left:0;right:0;bottom:0;height:8px;background:#000;z-index:100000}
#vwMark.on{background:#fff}
`;

const PAGE_JS = () => {
  const d = document;
  const mk = (tag, id) => { const e = d.createElement(tag); e.id = id; d.body.appendChild(e); return e; };
  const cap = mk('div', 'vwCap'), chip = mk('div', 'vwChip'), card = mk('div', 'vwCard'), cur = mk('div', 'vwCur');
  const mark = mk('div', 'vwMark');
  card.innerHTML = '<h1></h1><p></p><div></div>';
  window.__vw = {
    mark() { mark.classList.add('on'); setTimeout(() => mark.classList.remove('on'), 280); },
    cap(html) { cap.innerHTML = '<span>' + html + '</span>'; cap.classList.add('on'); this.mark(); },
    capOff() { cap.classList.remove('on'); },
    chip(t) { chip.textContent = t; chip.classList.toggle('on', !!t); },
    card(title, sub, list) {
      card.querySelector('h1').innerHTML = title || '';
      card.querySelector('p').innerHTML = sub || '';
      card.querySelector('div').innerHTML = list ? '<ul>' + list.map(x => '<li>' + x + '</li>').join('') + '</ul>' : '';
      card.classList.add('on');
      this.mark();
    },
    cardOff() { card.classList.remove('on'); },
    cursor(x, y) { cur.style.left = x + 'px'; cur.style.top = y + 'px'; cur.classList.add('on'); },
    tap() { cur.classList.add('tap'); setTimeout(() => cur.classList.remove('tap'), 160); },
    scroller() {
      const c = [d.querySelector('main'), d.scrollingElement, d.body];
      return c.find(e => e && e.scrollHeight - e.clientHeight > 40) || d.scrollingElement;
    },
    scrollBy(px) { this.scroller().scrollBy({ top: px, behavior: 'smooth' }); },
    scrollTop() { this.scroller().scrollTo({ top: 0, behavior: 'smooth' }); },
  };
};

// ---------------------------------------------------------------- driver
let page;
const missingSel = [];

const wait = ms => page.waitForTimeout(ms * SPEED);
const chip = t => page.evaluate(x => window.__vw.chip(x), t);

let T0 = 0;
function logCap(text) { CAPS.push({ i: CAPS.filter(c => c.text).length, key: speechKey(text), t: T0 ? Date.now() - T0 : 0, text }); }
async function say(text, extra = 0) {
  logCap(text);
  if (DUMP) return;
  await page.evaluate(t => window.__vw.cap(t), text);
  await wait(dur(text) / SPEED + extra);   // capDur() = clip length + a breath when narrated
}
async function card(title, sub, list, hold = 4200) {
  const speech = [title, sub].concat(list || []).filter(Boolean).join('. ').replace(/\.\./g, '.');
  logCap(speech);
  hold = Math.max(hold, capDur(speech));
  if (DUMP) return;
  await page.evaluate(([a, b, c]) => window.__vw.card(a, b, c), [title, sub, list || null]);
  await wait(hold);
  await page.evaluate(() => window.__vw.cardOff());
  await wait(600);
}
async function find(sel, nth = 0) {
  const els = await page.$$(sel);
  if (!els[nth]) { missingSel.push(sel + (nth ? '#' + nth : '')); console.log('  !! missing', sel, nth); return null; }
  return els[nth];
}
// keeps a target out from under the sticky top bar and off the caption strip
async function ensureVisible(el) {
  await el.scrollIntoViewIfNeeded().catch(() => {});
  for (let i = 0; i < 3; i++) {
    const b = await el.boundingBox().catch(() => null);
    if (!b) return null;
    const y = b.y + b.height / 2;
    if (y > 120 && y < 790) return b;
    await page.evaluate(d => window.__vw.scroller().scrollBy({ top: d, behavior: 'auto' }), y <= 120 ? y - 170 : y - 660);
    await page.waitForTimeout(320);
  }
  return await el.boundingBox().catch(() => null);
}
async function point(el) {
  const b = await ensureVisible(el);
  await wait(200);
  if (!b) return null;
  const x = b.x + b.width / 2, y = b.y + b.height / 2;
  await page.evaluate(([a, c]) => window.__vw.cursor(a, c), [x, y]);
  await el.evaluate(n => { n.classList.add('vw-ring'); setTimeout(() => n.classList.remove('vw-ring'), 2400); });
  await wait(420);
  return { x, y };
}
async function click(sel, text, { nth = 0, after = 900, hold = 0.55, expect = null } = {}) {
  if (text) logCap(text);
  if (DUMP) return;
  if (text) await page.evaluate(t => window.__vw.cap(t), text);
  const el = await find(sel, nth);
  if (!el) { if (text) await wait(dur(text) / SPEED); return; }
  await point(el);                                    // cursor + highlight ring
  const total = text ? dur(text) / SPEED : 0;
  if (text) await wait(total * hold);
  // re-measure: a smooth scroll during the caption may have moved the element
  const b = await ensureVisible(el);
  await page.evaluate(() => window.__vw.tap());
  if (b) {
    const x = b.x + b.width / 2, y = b.y + b.height / 2;
    if (y > 110 && y < 800) {
      await page.evaluate(([a, c]) => window.__vw.cursor(a, c), [x, y]);
      await page.mouse.click(x, y).catch(() => {});
    } else await el.click({ force: true }).catch(() => {});
  } else await el.click({ force: true }).catch(() => {});
  if (text) await wait(total * (1 - hold));   // let the sentence finish before moving on
  await wait(after);
  if (expect) {
    const ok = await page.waitForFunction(
      e => (document.querySelector(e.sel)?.textContent || '').includes(e.txt),
      expect, { timeout: 2500 }).then(() => true).catch(() => false);
    if (!ok) { console.log('  !! retrying click', sel, '->', expect.txt); await el.click({ force: true }).catch(() => {}); await wait(900); }
  }
}
async function hover(sel, text, { nth = 0, after = 1500 } = {}) {
  if (text) logCap(text);
  if (DUMP) return;
  if (text) await page.evaluate(t => window.__vw.cap(t), text);
  const el = await find(sel, nth);
  if (!el) { if (text) await wait(dur(text) / SPEED); return; }
  const p = await point(el);
  if (p) await page.mouse.move(p.x, Math.min(p.y, 860));
  const rest = text ? Math.max(after, dur(text) / SPEED - 1200) : after;
  await wait(rest);
}
async function set(sel, value, text) {
  if (text) logCap(text);
  if (DUMP) return;
  if (text) await page.evaluate(t => window.__vw.cap(t), text);
  const el = await find(sel);
  if (!el) { if (text) await wait(dur(text) / SPEED); return; }
  await point(el);
  const tag = await el.evaluate(n => n.tagName + ':' + (n.type || ''));
  if (tag.startsWith('SELECT')) await page.selectOption(sel, String(value)).catch(() => {});
  else if (tag.includes('range') || tag.includes('number') || tag.includes('text'))
    await el.evaluate((n, v) => { n.value = v; n.dispatchEvent(new Event('input', { bubbles: true })); n.dispatchEvent(new Event('change', { bubbles: true })); }, String(value));
  else if (tag.includes('checkbox')) await el.evaluate(n => n.click());
  if (text) await wait(Math.max(700, dur(text) / SPEED - 1400));
  await wait(400);
}
async function go(screen, text) {
  if (DUMP) { if (text) logCap(text); return; }
  await click(`.rail button[data-go="${screen}"], nav button[data-go="${screen}"], button[data-go="${screen}"]`, text, { after: 1400 });
  await page.evaluate(() => window.__vw.scrollTop());
  await wait(500);
}
async function read(px, text) {
  if (text) logCap(text);
  if (DUMP) return;
  if (text) await page.evaluate(t => window.__vw.cap(t), text);
  await page.evaluate(p => window.__vw.scrollBy(p), px);
  await wait(text ? dur(text) / SPEED : 1500);
}
const busy = async (ms = 25000, sel = null) => {
  await page.waitForTimeout(700);                                   // let the spinner appear
  await page.waitForFunction(() => !document.querySelector('.spin'), null, { timeout: ms }).catch(() => {});
  if (sel) await page.waitForFunction(
    s => { const e = document.querySelector(s); return e && !e.classList.contains('empty') && (e.children.length > 0 || e.textContent.trim().length > 3); },
    sel, { timeout: ms }).catch(() => { console.log('  !! never filled', sel); });
  await page.waitForTimeout(900);
};

// ---------------------------------------------------------------- chapters
const CH = {};

CH[0] = async () => {
  await chip('');
  await card('CLUTCH <span>IQ</span> v4',
    'Adaptive matchmaking, and a reinforcement-learning laboratory built on the same environment',
    ['One 48-state MDP studied with bandits, dynamic programming, Monte Carlo and Q-learning',
     'Every screen, tab, button and slider in this walkthrough',
     'Real data, simulated data and model estimates are labelled apart throughout'], 7000);
};

CH[1] = async () => {
  await chip('the shell');
  await say('Everything runs in one page. The <b>left rail</b> groups the screens: Mission control, Operations, the RL control room, the RL lab, and the Academic section.');
  await hover('.rail button[data-go="ops-data"]', 'Operations covers the product side: data, the live queue, tournaments and burnout.', { after: 1100 });
  await hover('.rail button[data-go="ctl-mdp"]', 'The control room and the lab are the RL screens. Each carries its syllabus unit as a badge: U1, U2, U3, U4, U5.', { after: 1100 });
  await hover('#engineBadge', 'Top right, the <b>engine badge</b> says who computed what you see. Right now it reads "Python backend", so every number comes from FastAPI.');
  await click('#engineBtn', 'The <b>Backend settings</b> button opens that choice.');
  await say('<b>Connect</b> re-probes the API. <b>Use browser engine</b> switches to the JavaScript port, which reproduces every RL endpoint offline with the same seeds. I will come back to that at the end.');
  await click('#engineClose', 'Close, and stay on the Python backend.');
  await hover('label.switch', '<b>Viva mode</b> adds the unit, the formula, the assumptions and a live worked example to every screen. I will turn it on later so the screens stay readable first.', { after: 900 });
};

CH[2] = async () => {
  await go('overview', 'Start at <b>Mission control</b>.');
  await chip('mission control');
  await say('The top panel animates the <b>agent-environment loop</b> from Unit 1: the agent reads a state, picks an action, the environment returns a reward and the next state.');
  await wait(2600);
  await hover('#loopSvg', 'The labels are live. Action, reward and next state update every cycle, and the whole loop is marked <i>Simulated</i> because the player reaction comes from our simulator.', { after: 2600 });
  await read(420, 'Below it, the production policy map: 48 states as four fatigue blocks, tilt down the rows, streak across the columns.');
  await say('Each cell is one state. The arrow is the action the shipped policy takes there: <b>down for LOWER</b> the lobby difficulty, <b>dot for KEEP</b>, <b>up for RAISE</b>.');
  await hover('#ovMap .cell', 'Hovering a cell names the state, its index and its key, so a cell is never just a colour.', { nth: 30, after: 2200 });
  await read(420, 'The tiles underneath are the headline numbers from the seeded experiment suite, not hand-typed figures.');
  await read(380, 'Two shortcuts finish the screen: <b>Open live session</b> and <b>See the formal MDP</b>.');
};

CH[3] = async () => {
  await go('ops-data', 'Operations starts with <b>Data and upload</b>.');
  await chip('operations · data');
  await say('This screen takes a real match-history CSV. The adapter does not expect one game: it detects which column plays which role by name and by statistics.');
  const f = await find('#csvFile');
  if (f) { await point(f); await f.setInputFiles(path.join(DATA, 'match_history_demo.csv')); }
  await say('I am attaching the demo history: 16 players, ten days, 865 matches. It is <i>simulated</i> data with invented gamertags, so nothing here describes a real player.');
  await click('#btnUpload', 'Upload and detect schema.');
  await busy(40000, '#engOut');
  await say('It found the roles: player, date, result, kills, deaths, assists, score, map. Then it engineered the features and clustered the players into labelled chemistry groups.');
  await read(360, 'The notes below are the <b>provenance</b> of what it just built, and they are deliberately blunt.');
  await say('It says sessions came from real dates, closeness fell back to the neutral 0.5 because this file has no scorelines, and rage quits are a <i>proxy</i>, because a match history does not log them.');
  await read(400, 'Next: <b>Estimate V(s) from uploaded sessions</b> — Monte Carlo prediction on the real sessions you just uploaded.');
  await click('#btnRealMc', 'Run it.');
  await busy(40000, '#realMcOut');
  await say('It replays each real session as an episode of bucketed states and averages the returns. The map is first-visit V(s); blank cells were never observed in this data.');
  await read(430, 'Control and off-policy evaluation are <b>not</b> offered here. A match history records no matchmaking action, so there is nothing to learn a policy from, and I will not invent one.');
  await say('That is what the second uploader is for: logs that <i>do</i> carry the action taken and the probability of taking it.');
  const lf = await find('#logFile');
  if (lf) { await point(lf); await lf.setInputFiles(path.join(DATA, 'rl_session_logs_demo_SIMULATED.csv')); }
  await click('#btnLogUpload', 'Uploading the demo session logs.');
  await busy(30000, '#logOut');
  await say('Now it reports the episodes, the steps, and which methods those logs unlock — including importance sampling, because behaviour probabilities are present.');
};

CH[4] = async () => {
  await go('ops-queue', 'Next, <b>Live queue and matchmaking</b>.');
  await chip('operations · queue');
  await say('The queue is a seeded simulation of players waiting. Each card shows rank, the archetype, and the derived fatigue and tilt.');
  await click('#qRefresh', '<b>New simulated queue</b> draws a fresh seeded lobby.');
  await busy(30000, '#qGrid');
  await click('#qMatch', '<b>Assemble balanced match</b> builds two teams.');
  await busy(30000, '#qTeams');
  await say('It balances skill and chemistry across the two sides, then reports the gap it achieved and what it traded off.');
  await read(430, 'The RL layer appears per player: the recommended difficulty for the state that player is in right now.');
  await read(400, 'The legend at the bottom is the honesty key used across the whole app: real data, simulated, model estimate, derived feature, and fallback or proxy.');
};

CH[5] = async () => {
  await go('ops-tournament', 'The <b>Tournament</b> screen takes two CSVs: maps, and players with their statistics.');
  await chip('operations · tournament');
  const m = await find('#tMaps'), p = await find('#tPlayers');
  if (m) { await point(m); await m.setInputFiles(path.join(DATA, 'tournament_maps_demo.csv')); }
  if (p) { await point(p); await p.setInputFiles(path.join(DATA, 'tournament_players_demo.csv')); }
  await say('I am using the demo pair: 8 maps and 24 players. <b>Use sample CSVs</b> loads the bundled files instead, if you have none.');
  await click('#tBuild', '<b>Build tournament</b>.');
  await busy(60000, '#tOut');
  await say('Archetypes come from the statistics, not from names: aggression, support, sniping and objective play, with a flex label when no role wins by a clear margin.');
  await read(420, 'It snake-drafts balanced teams, benches whoever is left over, and schedules the matches.');
  await read(420, 'For each match it recommends a map from a <b>fit score</b> — size, layout, and the roster archetypes. That score is a rule, so it is labelled a derived feature.');
  await read(380, 'And this is where Unit 2 enters the product: <b>Learn map with a bandit</b> turns the candidate maps into bandit arms.');
  await click('button:has-text("Learn map with a bandit")', 'Click it.', { after: 1600 });
  await busy(60000);
  await say('The map choice does not change the player\'s next state, so there is nothing to plan: it is a one-step decision, which is exactly what a bandit is for.');
};

CH[6] = async () => {
  await chip('operations · burnout');
  await go('ops-data', '<b>Burnout and coach</b> closes the operations group — and it needs the real players, so back to the data screen for a moment.');
  const bf = await find('#csvFile');
  if (bf) { await point(bf); await bf.setInputFiles(path.join(DATA, 'match_history_demo.csv')); }
  await click('#btnUpload', 'The simulated queue replaced the real players a moment ago, so I am re-running the upload to put them back in the pool.');
  await busy(40000, '#engOut');
  await go('ops-burnout', 'Now the burnout screen.');
  await set('#bPlayer', await page.evaluate(() => document.querySelector('#bPlayer option:nth-child(2)')?.value || ''), 'Pick a player with enough history.');
  await click('#bRun', '<b>Run forecast and report</b>.');
  await busy(30000, '#bF');
  await say('It fits the player\'s tilt trend across sessions and projects the next five matches, with the danger line drawn at 0.65.');
  await read(430, 'Then the coach report: what the derived features say, in sentences. It is generated from the trend, so it is a derived feature, not a diagnosis.');
};

CH[7] = async () => {
  await go('ctl-session', 'Into the RL control room. <b>Live session</b> is Unit 1 made concrete.');
  await chip('control room · live session');
  await say('You set the starting state yourself: fatigue, tilt and streak. These three buckets <i>are</i> the state.');
  await set('#sF', '2', 'Fatigue: TIRED.');
  await set('#sT', '2', 'Tilt: TILTED.');
  await set('#sK', '0', 'Streak: LOSING. A rough spot to be in.');
  await set('#sPol', 'dp_optimal', 'The policy dropdown picks who decides: here the DP-optimal policy.');
  await set('#sSeed', '7', 'The seed fixes the random numbers, so this session is reproducible.');
  await click('#sStep', '<b>Play one match</b>, one step of the loop.');
  await busy(8000);
  await say('The tape records the step: state, the action the policy chose, the reward, and the state it moved to. The gauges move with it.');
  await click('#sStep', 'Another match.');
  await wait(1400);
  await click('#sAuto', '<b>Auto-play</b> runs the rest of the session to a terminal state: a rage quit, a boredom quit, or simple exhaustion.');
  await busy(40000, '#sSummary');
  await say('When the episode ends, the return is computed <b>backwards</b> from the last reward: G<sub>t</sub> = R<sub>t+1</sub> + γ G<sub>t+1</sub>. That is the definition, applied to the episode you just watched.');
  await read(450, 'The curve plots G-nought for this same episode against γ, so you can see what discounting does.');
  await set('#sGamma', '0.2', 'Drag γ down to 0.2 and the agent becomes myopic: only the next match or two counts.');
  await set('#sGamma', '0.99', 'Push it to 0.99 and the whole session counts, which is why a far-sighted policy avoids rage quits.');
  await click('#sNewSeed', '<b>New seed</b> gives a different episode; <b>Reset</b> returns to the chosen start state.');
  await wait(900);
};

CH[8] = async () => {
  await go('ctl-q', '<b>Q-learning</b>. This is the screen that was mislabelled a DQN in version 3.');
  await chip('control room · q-learning');
  await say('It is a <b>table</b>: 49 states by 3 actions. With 48 discrete states there is nothing to approximate, so no neural network is involved — and the app now says so everywhere.');
  await read(400, 'The top map is the production policy: greedy with respect to the stored Q-table.');
  await read(420, 'Below it, the <b>temporal-difference explainer</b>. Pick a state, an action, a step size and a seed.');
  await click('#qMap .cell', 'The state is chosen by clicking it on the map — here a TIRED, TILTED, LOSING player.', { nth: 26, after: 1300 });
  await set('#qTdA', '2', 'Action RAISE.');
  await set('#qTdAlpha', '0.1', 'Step size α of 0.1, so the move is visible.');
  await click('#qTdRun', '<b>Sample a transition and update</b>.');
  await busy(30000, '#qTdOut');
  await say('It samples one real transition and shows the arithmetic: the reward, the bootstrapped target R + γ max Q(S\',·), the TD error, and the new value.');
  await say('The v3 bug lived right here. It bootstrapped from the next state even after a rage quit ended the episode. After a terminal step the target is just the reward, and that is what runs now.');
  await click('#qTabs button:has-text("Training lab")', 'The second tab is the <b>training lab</b>.', { expect: { sel: '#qPane1', txt: 'Training lab' } });
  await wait(900);
  await say('Every hyperparameter is exposed: episodes, α, γ, the ε schedule from 1.0 down to 0.05, the seed, and a state to watch while it learns.');
  await set('#qEp', '8000', 'I will train 8,000 episodes to keep this short. The shipped table used 30,000.');
  await hover('#qSave', 'This checkbox is off by default and matters: training here does <b>not</b> overwrite the production policy unless you explicitly ask it to.', { after: 1200 });
  await click('#qTrain', '<b>Train fresh table</b>.');
  await busy(180000, '#qTKpis');
  await say('The curves are from the run that just happened: the return per episode, the ε schedule decaying, and the Q-values of the watched state settling.');
  await read(450, 'The map at the end is the policy this run learned, with the states where it disagrees with the production table marked.');
};

CH[9] = async () => {
  await go('ctl-mdp', '<b>MDP explorer</b>: Unit 3, the formal model.');
  await chip('control room · mdp');
  await say('The state space is fatigue 4, tilt 4, streak 3 — 48 states, plus an absorbing END state, with s = 12f + 3t + k.');
  await say('The transition probabilities are <b>not</b> read off the simulator\'s formulas. They are estimated by rolling the simulator out 1,000 times per state-action pair: 144,000 simulated transitions, labelled a <i>model estimate</i>.');
  await set('#mSamples', '1000', 'That sample count is a control, so you can see the estimate get sharper as it grows.');
  await set('#mPol', 'dp_optimal', 'Pick the policy to inspect.');
  await click('#mMap .cell', 'Click any state to open it.', { nth: 26, after: 1500 });
  await busy(30000, '#mDetail');
  await say('Now the Bellman backup for that state, term by term, with real numbers: every next state, its estimated probability, the reward, and the discounted value it contributes.');
  await read(430, 'The sum of that column <i>is</i> Q(s,a). Take the max over actions and you have the optimality equation, not as a formula but as arithmetic you can check.');
  await click('#mActSeg button:has-text("RAISE")', 'Switch the action to RAISE and the whole breakdown recomputes.', { after: 1600 });
  await read(450, 'The graph underneath draws where that action actually sends you, ranked by probability.');
  await read(450, 'Fixing a policy collapses the MDP into a <b>Markov chain</b>, and the reward into an MRP. The app solves it in closed form, v = (I − γP)⁻¹R, and agrees with iterative evaluation to about 1e-8.');
  await set('#mGamma', '0.5', 'γ is live here too.');
  await read(400, 'Last, the <b>model quality</b> panel: how far the estimated probabilities sit from the simulator\'s exact law. We can only measure that because we wrote the simulator.');
};

CH[10] = async () => {
  await go('ctl-dp', '<b>Dynamic programming</b>: Unit 4, four tabs.');
  await chip('control room · dp');
  await say('All four need the model, which is what makes them model-based. They run in milliseconds on 48 states.');
  await click('#dTabs button:has-text("Policy evaluation")', 'First, <b>policy evaluation</b>.', { expect: { sel: '#dRun', txt: 'Evaluate' } });
  await click('#dRun', 'Run it: sweep the Bellman expectation backup until the largest change falls below θ.', { after: 1200 });
  await busy(60000, '#dPeK');
  await say('It reports the sweeps, the final residual, and the value map for the policy you evaluated. The closed-form solution is shown beside it as a cross-check.');
  await click('#dTabs button:has-text("Policy iteration")', 'Second, <b>policy iteration</b>.', { expect: { sel: '#dRun', txt: 'Run policy iteration' } });
  await click('#dRun', 'Evaluate, improve, repeat.', { after: 1200 });
  await busy(60000, '#dPiK');
  await say('Three improvement steps and it is policy-stable. The states that changed on each pass are outlined, so you can watch the policy actually move.');
  await read(430, 'Improvement never lowers the value of any state. That is the policy improvement theorem, and it is asserted in the test suite.');
  await click('#dTabs button:has-text("Value iteration")', 'Third, <b>value iteration</b>.', { expect: { sel: '#dRun', txt: 'Run value iteration' } });
  await click('#dRun', 'One optimality backup per sweep.', { after: 1200 });
  await busy(60000, '#dViK');
  await say('Forty-two sweeps to a tolerance of 1e-6 at γ = 0.9 — the experiments table quotes 52 for the same γ because it uses a tighter 1e-8. The interesting column is the ratio of successive changes: it never exceeds γ.');
  await read(430, 'That is the contraction property — the Banach fixed-point argument — measured rather than asserted. The sequence is Cauchy, so it converges to a unique fixed point.');
  await read(400, 'It also compares its answer with policy iteration. Same policy, different route.');
  await click('#dTabs button:has-text("γ sweep")', 'Fourth, the <b>γ sweep</b>.', { expect: { sel: '#dRun', txt: 'Run γ sweep' } });
  await click('#dRun', 'Solve the MDP across a range of discount factors.', { after: 1200 });
  await busy(90000, '#dGs');
  await say('This is the clearest result in the project: as γ rises, the optimal policy lowers difficulty more often and the rage-quit rate falls. Patience is worth something, and here is how much.');
};

CH[11] = async () => {
  await go('lab-bandit', 'Into the RL lab. <b>Bandit arena</b>: Unit 2.');
  await chip('rl lab · bandit arena');
  await say('The arms are maps. Their true values come from the tournament fit score rescaled, which is a rule — so the arm means are a <i>derived feature</i> and the rewards are <i>simulated</i>.');
  await set('#bSrc', 'custom', 'You can also type arm means directly, which is how the lecture examples are reproduced.');
  await set('#bSrc', 'sample_maps', 'Back to the map arms.');
  await set('#bK', '6', 'The slider sets how many arms are in play.');
  await say('Then the algorithms, each a checkbox: explore-only, greedy, explore-then-greedy, ε-greedy, optimistic initial values, and UCB.');
  await set('#bEps', '0.1', 'ε of 0.1 for ε-greedy.');
  await set('#bC', '1.414', 'c for UCB, the exploration weight in c√(ln t / N).');
  await set('#bQ0', '1', 'And the optimistic starting value: make every arm look good until it has been tried.');
  await set('#bRounds', '1000', '1,000 rounds per run.');
  await set('#bRuns', '50', 'Averaged over 50 runs, all algorithms on identical seeded environments.');
  await set('#bModel', 'bernoulli', 'Rewards can be Bernoulli thumbs-up or Gaussian quality.');
  await set('#bNs', 'patch', 'And this is the interesting one: a <b>balance patch</b> permutes the arm values halfway through.');
  await click('#bRun', '<b>Run arena</b>.');
  await busy(180000, '#bcReg');
  await say('Regret over time, and the share of pulls that were optimal. Greedy locks onto a mediocre map early. After the patch, the sample-average methods are stuck with their history.');
  await say('That is the argument for a <b>constant step size</b>: it weights recent rewards exponentially, so it forgets the pre-patch world.');
  await set('#bAlpha', '0.1', 'Set α to 0.1 and re-run to see that.');
  await click('#bRun', 'Run again.');
  await busy(180000, '#bcReg');
  await read(430, 'The inspector below opens one algorithm and replays its decisions.');
  await set('#bAlg', 'ucb', 'Pick UCB.');
  await set('#biSl', '40', 'And step through the rounds.');
  await say('For each round it shows the counts, the mean estimates, the confidence term, and which arm the rule therefore picks — the whole UCB formula, instantiated.');
  await read(420, 'Regret is reported two ways: the expected regret against the best arm, and the realised regret the lecture notes define.');
};

CH[12] = async () => {
  await go('lab-mc', '<b>Monte Carlo lab</b>: Unit 5, three tabs.');
  await chip('rl lab · monte carlo');
  await say('Monte Carlo needs no model at all. It averages complete-episode returns, which is why every episode here has to terminate — and ours always does, because fatigue only rises.');
  await set('#pPol', 'dp_optimal', 'Prediction first: choose the policy to evaluate.');
  await set('#pN', '3000', '3,000 episodes.');
  await set('#pStart', 'uniform', 'Starting states spread uniformly so every state gets visited.');
  await click('#pRun', '<b>Run MC prediction</b>.');
  await busy(180000, '#pV');
  await say('Two estimates side by side: <b>first-visit</b> uses the first time a state appears in an episode, <b>every-visit</b> uses all of them.');
  await read(430, 'And the error against the DP answer as episodes accumulate. It falls roughly as one over the square root of the visits — experience buys accuracy, slowly.');
  await click('#mcTabs button:has-text("Control")', 'Second tab: <b>control</b>, learning a policy rather than evaluating one.', { expect: { sel: '#mcPane', txt: 'Monte Carlo control' } });
  await wait(900);
  await busy(180000, '#cK');
  await say('It opens on <b>on-policy ε-soft</b>, which keeps exploring forever — so it converges to the best ε-soft policy, not the greedy optimum. That is why it lands near 97% rather than 100%.');
  await set('#cN', '10000', '10,000 episodes.');
  await click('#cVar button[data-v="es"]', '<b>Exploring starts</b> is legitimate here because our simulator can be placed in any state, which is rare and worth saying out loud.', { after: 1500 });
  await busy(180000, '#cK');
  await say('Every episode begins from a random state-action pair, so coverage is free and this variant gets closest to the optimum.');
  await click('#cVar button[data-v="off_policy"]', '<b>Off-policy with weighted importance sampling</b> learns a greedy target from an exploring behaviour policy.', { after: 1500 });
  await busy(180000, '#cK');
  await say('It only learns from the tail of each episode, after the last non-greedy action — which is why it trails the other two.');
  await click('#mcTabs button:has-text("Off-policy")', 'Third tab: <b>off-policy evaluation</b>, the practical question.', { expect: { sel: '#mcPane', txt: 'Off-policy evaluation' } });
  await wait(900);
  await say('"Would a new matchmaker be better?" — answered from sessions logged under the current one, before deploying anything.');
  await say('The ratio ρ is the product of π over b along the episode. <b>Ordinary</b> importance sampling is unbiased but high-variance; <b>weighted</b> is biased but stable; per-decision and discounting-aware cut the variance further.');
  await set('#oPre', 'small', 'Scenario one: a <b>small change</b> from the logged policy.');
  await click('#oRun', 'Estimate the target value.', { after: 1200 });
  await busy(180000, '#oTbl');
  await say('All four estimators, the truth to compare against, and the <b>effective sample size</b>. Weighted IS lands within a hundredth of the truth, with an ESS in the thousands.');
  await set('#oPre', 'large', 'Now the honest failure: <b>deploy the DP-optimal policy</b>, a large change from what was logged.');
  await click('#oRun', 'Estimate again.', { after: 1200 });
  await busy(180000, '#oTbl');
  await say('The effective sample size collapses to a few dozen, and the app <b>flags the estimate unreliable</b> instead of quoting it. A few episodes carry all the weight — that is the variance problem, not a bug.');
  await read(400, 'Coverage is checked too. If the target policy would take an action the logged policy never takes, the estimate is refused outright.');
};

CH[13] = async () => {
  await go('lab-compare', '<b>Algorithm comparison</b>.');
  await chip('rl lab · comparison');
  await say('Everything that solves the session problem, scored on identical seeded episodes. Bandits are deliberately absent: a one-step map choice does not belong on this leaderboard.');
  await set('#xN', '500', '500 evaluation episodes.');
  await set('#xStart', 'session_start', 'Started where a real session starts.');
  await click('#xRun', '<b>Run comparison</b>.');
  await busy(240000, '#xL');
  await say('The leaderboard with confidence intervals. Because every policy faces the same random numbers, the <b>paired</b> differences are far tighter than comparing independent means.');
  await read(430, 'DP-optimal against always-KEEP: the discounted return goes from 8.53 to 11.10, and rage-quit sessions fall from 45% to 29%.');
  await read(430, 'The agreement matrix shows where the methods pick different actions. Q-learning matches DP on 79% of states.');
  await read(450, 'And the panel explains the disagreements rather than hiding them: they cluster where two actions have nearly equal value, or in states the learner rarely visited.');
};

CH[14] = async () => {
  await go('acad-viva', 'The Academic group. <b>Viva questions</b> first.');
  await chip('academic');
  await say('Thirty likely questions with short defensible answers, each linked to the screen that demonstrates it.');
  await read(500, '');
  await click('button:has-text("Show it live")', '<b>Show it live</b> jumps to the screen that proves the answer.', { nth: 3, after: 1800 });
  await go('acad-syllabus', 'Next, the <b>syllabus mapping</b>.');
  await say('Every topic in Units 1 to 5, what implements it, where to demonstrate it, and an honest status: implemented, demonstrated, explained, or not covered.');
  await click('button:has-text("Unit 4")', 'Filter by unit.', { after: 1500 });
  await read(450, 'Nothing is claimed as implemented unless the code is in the repository and the tests exercise it. Thompson sampling and DQN appear here as <b>not covered</b>.');
  await go('acad-formulation', '<b>Mathematical formulation</b>: the MDP written out.');
  await read(480, 'The tuple, the state encoding, the reward with its three weighted terms, the Bellman equations, and the simulator assumptions stated as assumptions.');
  await read(480, 'The reward is the other thing v3 got wrong. Its session term was measured in minutes and swamped everything else, so a rage quit cost less than an ordinary bad match. It is now a bounded indicator.');
  await go('acad-numericals', '<b>Notes numericals</b>: the worked examples from the lectures, recomputed.');
  await read(480, 'The dice game, the four-by-four gridworld, the returns exercise, the exploration and UCB tables, the Monte Carlo A and B episodes.');
  await say('Fifteen of the lecture examples have a printed answer. All fifteen come out the same, and each one is also a unit test.');
  await go('acad-experiments', '<b>Experiment results</b>: the seeded suite, E1 through E8.');
  await read(480, 'Model accuracy against sample count, the DP solution, the γ study, Monte Carlo error against episodes, the method comparison, bandit regret, the non-stationary recovery, and importance-sampling reliability.');
  await say('These are read from results.json, which is written by running the suite. If a number is not in that file, it is not on this screen.');
  await go('acad-architecture', '<b>Architecture</b>: the modules and the request flow.');
  await read(450, 'One problem definition in rl_core, the model and the planners above it, FastAPI serving it, and a JavaScript port of the same engine for offline use.');
  await go('acad-limitations', 'And <b>Limitations and honesty</b>, which I would open first in a viva, not last.');
  await read(450, 'Hand-specified dynamics. A coarse 48-bucket state. Proxies for rage quits. An estimated model. Importance-sampling variance. Constant-α noise. Rule-based bandit arms. No live A/B test. In-memory storage.');
};

CH[15] = async () => {
  await chip('viva mode');
  await click('label.switch input#vivaToggle', 'Now <b>Viva mode</b>, the switch from the beginning.');
  await go('ctl-dp', 'Back to dynamic programming with it on.');
  await say('Every screen now carries its unit, the concept, why it is used here, the formula, the inputs and outputs, the assumptions, a worked example from the current run, and the file and function that compute it.');
  await read(480, 'It is the same panel on all nineteen screens, so there is no screen you can be asked about that cannot explain itself.');
  await click('label.switch input#vivaToggle', 'Switching it back off.');
  await chip('offline engine');
  await click('#engineBtn', 'Last thing: the engine switch.');
  await click('#engineOffline', '<b>Use browser engine</b>.');
  await wait(1200);
  await say('The badge changed. The Python backend is no longer being called — the JavaScript port is computing everything now, from the same seeds.');
  await go('ctl-dp', 'Value iteration again, offline.');
  await click('#dTabs button:has-text("Value iteration")', 'Back to the value-iteration tab — the screen remembers whichever tab you left open.', { expect: { sel: '#dRun', txt: 'Run value iteration' } });
  await click('#dRun', 'Run it.', { after: 1200 });
  await busy(90000, '#dViK');
  await say('Same sweeps, same values. The test suite checks that parity in both directions, which is what makes the single-file clutchiq.html honest: it is the real engine, not a screenshot.');
  await say('What you lose offline is only the CSV features — upload, the tournament, burnout, and logged episodes — because those need the server.');
};

CH[16] = async () => {
  await chip('');
  await card('How to run it',
    '',
    ['<b>pip install -r requirements.txt</b>',
     '<b>uvicorn backend_app:app --port 8000</b> &nbsp;→&nbsp; open http://localhost:8000/',
     'No server? Open <b>clutchiq.html</b> — every RL screen still works',
     '<b>pytest -q</b> &nbsp;→&nbsp; 130 tests, including Python-to-JavaScript parity',
     '<b>python rl_experiments.py</b> &nbsp;→&nbsp; regenerates every number in the app'], 9000);
  await card('What is real, and what is not',
    'The line to hold in a viva',
    ['<b>Real:</b> uploaded histories → features, clusters, matchmaking, burnout, MC prediction',
     '<b>Simulated:</b> player reactions to difficulty, every RL episode, bandit rewards',
     '<b>Model estimate:</b> the transition probabilities, from 144,000 rollouts',
     '<b>Proxy:</b> rage quits and closeness when the data does not log them',
     'The policy is learned on a documented simulator — not fitted to real players'], 9500);
};

// ---------------------------------------------------------------- main
(async () => {
  // ---- dump mode: run the chapters with a stubbed browser to print the narration
  if (DUMP) {
    const el = {
      scrollIntoViewIfNeeded: async () => {}, boundingBox: async () => ({ x: 10, y: 300, width: 60, height: 24 }),
      evaluate: async () => '', click: async () => {}, setInputFiles: async () => {},
    };
    page = {
      evaluate: async () => '', $: async () => el, $$: async () => [el, el, el, el, el, el],
      waitForTimeout: async () => {}, waitForFunction: async () => ({}), selectOption: async () => {},
      textContent: async () => '', mouse: { click: async () => {}, move: async () => {} },
    };
    for (const n of Object.keys(CH).map(Number).sort((a, b) => a - b)) {
      CAPS.push({ chapter: n });
      try { await CH[n](); } catch (e) { console.error('dump error ch' + n, String(e).slice(0, 120)); }
    }
    fs.writeFileSync(process.env.VW_DUMP, JSON.stringify(CAPS, null, 1));
    console.log('captions:', CAPS.filter(c => c.text).length, '->', process.env.VW_DUMP);
    return;
  }
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ['--force-device-scale-factor=1', '--hide-scrollbars'] });
  const ctx = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    recordVideo: { dir: OUT, size: { width: 1600, height: 900 } },
    reducedMotion: 'no-preference',
  });
  page = await ctx.newPage();

  if (FONTS) {
    const files = {};
    for (const [fam, dir, file] of [
      ['Chakra Petch', 'fontsource-chakra-petch', 'chakra-petch-latin-%W%-normal.woff2'],
      ['IBM Plex Sans', 'fontsource-ibm-plex-sans', 'ibm-plex-sans-latin-%W%-normal.woff2'],
      ['IBM Plex Mono', 'fontsource-ibm-plex-mono', 'ibm-plex-mono-latin-%W%-normal.woff2'],
    ]) for (const w of [400, 500, 600, 700]) {
      const p = path.join(FONTS, dir, 'files', file.replace('%W%', w));
      if (fs.existsSync(p)) files[`${fam}|${w}`] = p;
    }
    const css = Object.entries(files).map(([k, p]) => {
      const [fam, w] = k.split('|');
      return `@font-face{font-family:"${fam}";font-style:normal;font-weight:${w};font-display:block;src:url(https://fonts.gstatic.com/vwfont/${path.basename(p)}) format('woff2');}`;
    }).join('\n');
    await page.route('https://fonts.googleapis.com/**', r => r.fulfill({ status: 200, contentType: 'text/css', body: css }));
    await page.route('https://fonts.gstatic.com/vwfont/**', r => {
      const hit = Object.values(files).find(p => r.request().url().endsWith(path.basename(p)));
      return hit ? r.fulfill({ status: 200, contentType: 'font/woff2', body: fs.readFileSync(hit) }) : r.abort();
    });
  }

  const errs = [];
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 140)); });
  page.on('pageerror', e => errs.push('PAGEERROR ' + String(e).slice(0, 160)));

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  await page.addStyleTag({ content: CSS });
  await page.evaluate(PAGE_JS);
  await page.waitForTimeout(1200);
  console.log('engine:', await page.textContent('#engineText'));
  T0 = Date.now();

  if (process.env.VW_PRELUDE) {
    const [scr, px] = process.env.VW_PRELUDE.split(':');
    await page.evaluate(x => App.go(x), scr);
    await page.waitForTimeout(900);
    if (px) { await page.evaluate(v => window.__vw.scrollBy(+v), px); await page.waitForTimeout(1200); }
    await page.waitForTimeout(800);
  }
  const order = Object.keys(CH).map(Number).sort((a, b) => a - b).filter(n => !ONLY.length || ONLY.includes(n));
  for (const n of order) {
    const t0 = Date.now();
    console.log(`\n== chapter ${n}`);
    try { await CH[n](); } catch (e) { console.log('  chapter error:', String(e).slice(0, 200)); }
    console.log(`   ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await page.evaluate(() => { window.__vw.capOff(); window.__vw.chip(''); });
  await page.waitForTimeout(1500);

  console.log('\nmissing selectors:', missingSel.length ? missingSel : 'none');
  if (missing.length) console.log('captions with no voice clip:', missing.length, missing.slice(0, 5));
  console.log('console errors:', [...new Set(errs)].slice(0, 10));
  if (AUDIO) { fs.writeFileSync(path.join(OUT, 'captions.json'), JSON.stringify(CAPS.filter(c => c.text), null, 1)); console.log('caption log ->', path.join(OUT, 'captions.json')); }
  const v = page.video();
  await ctx.close();
  await browser.close();
  console.log('video:', await v.path());
})();
