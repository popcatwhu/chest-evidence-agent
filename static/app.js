'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
} [c]));
const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const modeNames = {
  verified: '临床证据对照与复核',
  tools: '工具分析',
  direct: '直接模型推理'
};
const modeDescriptions = {
  verified: '整合图像工具与医学资料，对照候选病因及支持、反对证据，再生成报告并复核引用与事实冲突。',
  tools: '使用图像工具与医学资料生成报告，跳过模型证据复核。',
  direct: '直接结合病史与原始胸片推理，作为工具流程的对照。'
};
const statusNames = {
  queued: '排队中',
  running: '分析中',
  completed: '已完成',
  failed: '执行失败',
  interrupted: '已中断'
};
const stageNames = {
  prepare: '整理病例',
  facts: '核对事实',
  observe: '观察胸片',
  radiology: '专用影像工具',
  plan: '分析计划',
  tools: '图像工具',
  retrieve: '检索资料',
  diagnose: '生成报告',
  contrast: '临床证据对照',
  verify: '证据复核',
  repair: '修订报告',
  update: '比较变化',
  complete: '分析完成',
  error: '执行失败'
};
const state = {
  cases: [],
  selected: null,
  mode: 'verified',
  health: null,
  connected: false,
  history: [],
  reportRun: null,
  activeRun: null,
  submitting: false,
  selecting: 0,
  file: null,
  blob: null,
  poll: null,
  polling: false,
  evaluation: null
};
const emptyReport = $('report-content').innerHTML;
let toastTimer;

function toast(text) {
  $('toast').textContent = text;
  $('toast').classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 3000);
}

function message(text = '', error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('hidden', !text);
  $('message').classList.toggle('error', error);
}

function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch (_) {}
}

function date(value, short = false) {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '时间未知';
  return d.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    ...(short ? {} : {
      year: 'numeric'
    })
  });
}

function href(value, artifact = false) {
  try {
    const u = new URL(value, location.origin);
    if (!['https:', 'http:'].includes(u.protocol)) return '';
    if (artifact && (u.origin !== location.origin || !u.pathname.startsWith('/artifacts/'))) return '';
    return u.href;
  } catch (_) {
    return '';
  }
}
async function api(path, options = {}) {
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), 20000);
  try {
    const response = await fetch(path, {
      ...options,
      signal: abort.signal
    });
    const raw = await response.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch (_) {
      throw Error(response.ok ? '服务返回了无法读取的内容，请刷新页面。' : `服务请求失败（${response.status}）`);
    }
    if (!response.ok) {
      const detail = data.detail;
      throw Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(x => x.msg).join('；') : `服务请求失败（${response.status}）`);
    }
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw Error('连接超时，请检查远程服务或端口转发。');
    throw e;
  } finally {
    clearTimeout(timeout);
  }
}

function updateControls() {
  const busy = !!state.activeRun || state.submitting;
  const ready = state.connected && state.health?.model_ready;
  $('analyze').disabled = !state.selected || !ready || busy;
  $('new-case').disabled = busy;
  $('top-new-case').disabled = busy;
  $('save-case').disabled = busy;
  $('save-supplement').disabled = busy;
  document.querySelectorAll('.case-item').forEach(b => b.disabled = busy);
  document.querySelectorAll('[data-mode]').forEach(b => b.disabled = busy);
  $('analyze').innerHTML = busy ? `<span>分析进行中</span><span class="spinner"></span>` : `<span>开始分析</span>${icon('arrow')}`;
  $('analyze-hint').textContent = busy ? '任务在远程运行，刷新页面后可恢复进度。' : !state.connected ? '正在连接远程服务…' : state.health?.inference_paused ? '模型评测中，新分析暂时暂停。' : !ready ? '模型尚未准备好，历史报告仍可查看。' : !state.selected ? '先保存一个病例，或从左侧选择。' : !state.health?.model_loaded ? '首次分析需要加载模型，耗时会较长。' : '模型已加载，可提交分析。';
}
async function health() {
  try {
    state.health = await api('/api/health');
    state.connected = true;
    const h = state.health;
    $('health-label').textContent = h.inference_paused ? '推理暂时暂停' : h.model_ready ? '服务已连接' : '模型尚未就绪';
    $('connection-dot').className = 'status-dot ' + (h.inference_paused || !h.model_ready ? 'paused' : 'online');
    $('model-label').textContent = h.model_name || (h.backend === 'api' ? 'API 模型' : '本地模型');
  } catch (_) {
    state.connected = false;
    $('health-label').textContent = '服务连接中断';
    $('connection-dot').className = 'status-dot offline';
    $('model-label').textContent = '正在尝试重新连接';
  }
  updateControls();
  if ($('connection-dialog').open) renderConnection();
}

function sidebar(open) {
  $('sidebar').classList.toggle('open', open);
  $('sidebar-scrim').hidden = !open;
  $('menu-toggle').setAttribute('aria-expanded', String(open));
}
async function view(name) {
  document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  $('workspace-view').classList.toggle('hidden', name !== 'workspace');
  $('evaluation-view').classList.toggle('hidden', name !== 'evaluation');
  $('breadcrumb-current').textContent = name === 'workspace' ? '诊断工作台' : '评测概览';
  sidebar(false);
  if (name === 'evaluation') await loadEvaluation();
}

function tab(name) {
  document.querySelectorAll('[data-tab]').forEach(b => {
    const active = b.dataset.tab === name;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', String(active));
    b.tabIndex = active ? 0 : -1;
  });
  for (const t of ['report', 'evidence', 'history']) $(`${t}-content`).classList.toggle('hidden', t !== name);
}

function setMode(mode) {
  if (!modeNames[mode]) return;
  state.mode = mode;
  document.querySelectorAll('[data-mode]').forEach(b => {
    const active = b.dataset.mode === mode;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  });
  $('mode-description').textContent = modeDescriptions[mode];
}

function caseTitle(c) {
  return c.title.replace(/^(?:Qwen|Lingshu)[^·]*临床开发回归\s*·\s*/, '');
}

function renderCases() {
  const query = $('case-search').value.trim().toLowerCase();
  const rows = state.cases.filter(c => `${c.title} ${c.context}`.toLowerCase().includes(query));
  $('case-count').textContent = state.cases.length;
  $('case-list').innerHTML = rows.length ? rows.map(c => {
    const subtitle = `${c.image_path ? '含胸片' : '仅病史'} · ${date(c.created,true)}`;
    return `<button class="case-item ${state.selected?.id === c.id ? 'selected' : ''}" data-case="${esc(c.id)}" title="${esc(c.title)}" ${state.activeRun || state.submitting ? 'disabled' : ''}>${icon(c.image_path ? 'image' : 'file')}<span class="case-item-text"><strong>${esc(caseTitle(c))}</strong><small>${esc(subtitle)}</small></span></button>`;
  }).join('') : `<div class="sidebar-empty">${query ? '没有找到匹配的病例。' : '还没有病例，点击上方 + 新建。'}</div>`;
}

function resetUpload() {
  if (state.blob) URL.revokeObjectURL(state.blob);
  state.blob = null;
  state.file = null;
  $('image-upload').value = '';
  $('upload-label').textContent = '点击上传，或拖入胸片';
  $('clear-upload').classList.add('hidden');
  if (!state.selected) $('image-section').classList.add('hidden');
}

function chooseFile(file) {
  if (!file) return;
  if (!(file.type ? ['image/png', 'image/jpeg'].includes(file.type) : /\.(png|jpe?g)$/i.test(file.name))) {
    message('请选择 PNG 或 JPEG 图片。', true);
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    message('图片超过 20 MB，请选择较小的文件。', true);
    return;
  }
  resetUpload();
  state.file = file;
  state.blob = URL.createObjectURL(file);
  $('preview').src = state.blob;
  $('image-section').classList.remove('hidden');
  $('image-caption').textContent = file.name;
  $('upload-label').textContent = file.name;
  $('clear-upload').classList.remove('hidden');
  message();
}

function clearReport() {
  state.reportRun = null;
  $('report-content').innerHTML = emptyReport;
  $('evidence-content').innerHTML = '<div class="simple-empty">分析完成后，这里会展示本次使用的证据。</div>';
  $('evidence-count').textContent = '0';
  $('download-report').classList.add('hidden');
}
async function selectCase(id, force = false) {
  if ((state.activeRun || state.submitting) && !force) {
    toast('请等待当前分析完成。');
    return;
  }
  const token = ++state.selecting;
  state.selected = state.cases.find(c => c.id === id) || null;
  const c = state.selected;
  resetUpload();
  $('image-section').classList.add('hidden');
  $('preview').removeAttribute('src');
  clearReport();
  message();
  state.history = [];
  $('history-count').textContent = '0';
  $('history-content').innerHTML = '<div class="simple-empty">' + (c ? '正在读取历史记录…' : '保存病例后即可查看历史分析记录。') + '</div>';
  $('progress-panel').classList.add('hidden');
  $('case-form').classList.toggle('hidden', !!c);
  $('case-detail').classList.toggle('hidden', !c);
  $('supplement-section').classList.toggle('hidden', !c);
  $('supplement-section').open = false;
  $('supplement').value = '';
  $('case-label').textContent = c ? '已保存' : '新病例';
  $('case-label').className = 'tag ' + (c ? 'green' : 'neutral');
  if (c) {
    $('selected-case-title').textContent = caseTitle(c);
    $('case-created').textContent = date(c.created);
    $('case-context').textContent = c.context;
    if (c.image_path) {
      $('preview').src = '/api/cases/' + encodeURIComponent(c.id) + '/image';
      $('image-section').classList.remove('hidden');
      $('image-caption').textContent = '原始胸片';
    }
    storage('chestSelectedCase', c.id);
  } else {
    storage('chestSelectedCase', null);
    $('case-form').reset();
  }
  const url = new URL(location.href);
  if (c) url.searchParams.set('case', c.id);
  else url.searchParams.delete('case');
  history.replaceState(null, '', url);
  renderCases();
  updateControls();
  tab('report');
  await view('workspace');
  if (!c) {
    $('case-title').focus({
      preventScroll: true
    });
    return;
  }
  try {
    const runs = await api('/api/cases/' + encodeURIComponent(c.id) + '/runs');
    if (token !== state.selecting) return;
    state.history = runs;
    renderHistory();
    const completed = runs.find(r => r.status === 'completed');
    if (completed) {
      const run = await api('/api/runs/' + encodeURIComponent(completed.id));
      if (token === state.selecting && run.result) renderRun(run);
    }
  } catch (e) {
    if (token === state.selecting) message(e.message, true);
  }
}

function refs(ids) {
  return (ids || []).map(id => `<button class="ref" data-ref="${esc(id)}" title="查看证据 ${esc(id)}">${esc(id)}</button>`).join('');
}

function claims(items, positive = false) {
  return (items || []).map(c => `<div class="claim">${positive ? icon('check') : ''}${esc(c.text)}${refs(c.evidence_ids)}</div>`).join('');
}

function list(items, empty = '未列出') {
  return items?.length ? `<ul class="plain-list">${items.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : `<p class="subtle-text">${esc(empty)}</p>`;
}

function evidenceBody(e) {
  if (e.status === 'failed') return `<p class="model-observation-note">工具执行失败，不能用于支持或排除诊断。</p><div class="evidence-text">${esc(e.content)}</div>`;
  try {
    const d = JSON.parse(e.content);
    if (e.id === 'I-classify' && d.scores) return `<p class="model-observation-note">模型分数，不是患者患病概率；低分不能排除疾病。</p>${Object.entries(d.scores).sort((a,b) => b[1]-a[1]).map(([key,value]) => `<div class="evidence-score-row"><span>${esc(key)}</span><span>${Number(value).toFixed(3)}</span></div>`).join('')}`;
    if (Array.isArray(d.visual_findings) || Array.isArray(d.observations)) return `<p class="model-observation-note">模型生成的观察，仍需原图和专业人员核对。</p>${list(d.visual_findings || d.observations)}<details class="facts-box"><summary>查看完整原始输出</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details>`;
    return `<pre class="evidence-text">${esc(JSON.stringify(d,null,2))}</pre>`;
  } catch (_) {
    return `<div class="evidence-text">${esc(e.content)}</div>`;
  }
}

function renderEvidence(evidence) {
  $('evidence-count').textContent = evidence.length;
  const groups = [
    ['case', '病例原文'],
    ['image', '影像与工具观察'],
    ['knowledge', '检索资料']
  ];
  const known = new Set(groups.map(x => x[0]));
  if (evidence.some(e => !known.has(e.kind))) groups.push(['other', '其他证据']);
  $('evidence-content').innerHTML = '<p class="evidence-intro">引用编号对应本次分析保存的证据快照。模型观察与公开资料均不能替代患者的确认检查。</p>' + groups.map(([kind, label]) => {
    const rows = evidence.filter(e => kind === 'other' ? !known.has(e.kind) : e.kind === kind);
    if (!rows.length) return '';
    return `<h3 class="evidence-group-heading">${label} · ${rows.length}</h3>` + rows.map(e => {
      const source = e.source ? href(e.source) : '';
      const artifact = e.artifact ? href(e.artifact, true) : '';
      const type = e.source_type || (e.kind === 'case' ? '本次分析的病例输入快照' : e.id === 'I-original' ? '原始胸片的直接视觉观察' : e.kind === 'image' ? '模型生成观察或工具结果' : '公开医学资料');
      return `<article class="evidence-item" id="evidence-${esc(e.id)}"><div class="evidence-item-heading"><span class="evidence-id">${esc(e.id)}</span><h3>${esc(e.title)}</h3>${e.status === 'failed' ? '<span class="tag red">执行失败</span>' : ''}</div><div class="evidence-type">${esc(type)}</div>${evidenceBody(e)}${source ? `<a class="source-link" href="${esc(source)}" target="_blank" rel="noopener noreferrer">查看原始来源 ↗</a>` : ''}${artifact ? `<img class="overlay" src="${esc(artifact)}" alt="工具生成的分割叠加结果" loading="lazy">` : ''}</article>`;
    }).join('');
  }).join('');
}

function renderRun(run) {
  if (!run.result?.report) return;
  state.reportRun = run;
  const z = run.result,
    r = z.report,
    v = z.verification || {},
    primary = r.most_likely || {},
    metrics = z.metrics || {};
  const assessed = r.assessment || '有限';
  const verifyName = v.status === 'passed_checks' ? '引用与本轮检查通过' : v.status === 'needs_review' ? '仍需复核' : '未执行模型复核';
  $('report-content').innerHTML = `<div class="report-meta"><span>${esc(date(run.created))} · ${esc(modeNames[run.mode] || run.mode)}</span><div><span class="tag ${assessed === '较充分' ? 'green' : 'amber'}">证据${esc(assessed)}</span>${r.answer_choice ? `<span class="tag neutral">选项 ${esc(r.answer_choice)}</span>` : ''}</div></div>${state.selected && run.context !== state.selected.context ? '<div class="inline-message" style="margin:0 0 15px">这份历史报告使用的资料与当前病例不同，请重新分析以更新判断。</div>' : ''}<article class="primary-diagnosis"><div class="diagnosis-kicker"><span>最可能的候选诊断</span><button id="copy-conclusion" class="icon-button" aria-label="复制候选诊断" title="复制候选诊断">${icon('copy')}</button></div><h2>${esc(primary.name || '未提供候选诊断')}</h2><p class="diagnosis-note">候选判断，需结合确认检查进一步评估</p><div class="claim-heading">支持证据</div>${claims(primary.support,true) || '<p class="subtle-text">未列出支持证据</p>'}${primary.against?.length ? `<div class="claim-heading negative">反对或矛盾证据</div>${claims(primary.against)}` : ''}</article><div class="section-title">鉴别诊断 <small>${(r.differentials || []).length} 个候选</small></div>${(r.differentials || []).map((c,i) => `<article class="differential"><div class="differential-heading"><span>${String(i+1).padStart(2,'0')}</span><h3>${esc(c.name)}</h3></div><div class="claim-heading">支持依据</div>${claims(c.support) || '<p class="subtle-text">未列出支持证据</p>'}${c.against?.length ? `<div class="claim-heading negative">反对或矛盾依据</div>${claims(c.against)}` : ''}</article>`).join('') || '<p class="subtle-text">本次未列出其他候选诊断</p>'}<div class="report-two-columns"><section class="report-block"><h3>需要补充的信息</h3>${list(r.missing_information,'报告未列出缺失信息')}</section><section class="report-block"><h3>进一步评估方向</h3>${r.recommended_checks?.length ? r.recommended_checks.map(c => `<div class="check-item"><strong>${esc(c.name)}</strong><p>${esc(c.purpose)}${refs(c.evidence_ids)}</p></div>`).join('') : list(r.next_checks,'本次未列出进一步检查')}</section></div>${r.findings?.length ? `<div class="section-title">其他观察</div>${claims(r.findings)}` : ''}<div class="change-note"><strong>本次判断变化</strong>${esc(r.change_summary || '首次分析')}</div><details class="review-box" ${v.status === 'needs_review' ? 'open' : ''}><summary>${icon('shield')}证据复核<span class="tag ${v.status === 'needs_review' ? 'amber' : v.status === 'passed_checks' ? 'green' : 'neutral'}">${verifyName}</span></summary>${list([...(v.issues || []),...(v.notes || [])],v.performed ? '本轮未保留待处理检查问题' : '本次未进行模型语义复核')}<p>${esc(v.note || '这些检查不能证明诊断正确。')}</p></details>${z.clinical_contrast?.key_case_clues?.length ? `<details class="facts-box"><summary>诊断中核对的关键病例原文 · ${z.clinical_contrast.key_case_clues.length} 项</summary>${z.clinical_contrast.key_case_clues.map(f => `<div class="fact-row"><strong>${esc(f.name)}</strong>：${esc(f.value)}${refs([f.source_id])}<div class="subtle-text">原文：${esc(f.source_quote)}</div></div>`).join('')}</details>` : ''}${z.profile?.facts?.length ? `<details class="facts-box"><summary>查看已核对的病例事实 · ${z.profile.facts.length} 项</summary>${z.profile.facts.map(f => `<div class="fact-row"><strong>${esc(f.name)}</strong>：${esc(f.value)}${refs([f.source_id])}</div>`).join('')}</details>` : ''}<div class="report-stats"><span>${icon('clock')}耗时 ${esc(metrics.elapsed_seconds ?? '—')} 秒</span><span>模型调用 ${(metrics.model_calls || []).length} 次</span><span>工具调用 ${esc(metrics.tool_calls ?? '—')} 次</span>${z.model ? `<span>${esc(z.model)}</span>` : ''}</div>`;
  renderEvidence(z.evidence || []);
  $('download-report').href = '/api/runs/' + encodeURIComponent(run.id) + '/download';
  $('download-report').classList.remove('hidden');
  renderHistory();
}

function renderHistory() {
  $('history-count').textContent = state.history.length;
  $('history-content').innerHTML = state.history.length ? '<p class="history-intro">每次分析保留当时的病例输入、证据与报告。点击记录查看对应版本。</p>' + state.history.map(run => `<button class="history-item ${state.reportRun?.id === run.id ? 'selected' : ''}" data-run="${esc(run.id)}"><div><strong>${esc(modeNames[run.mode] || run.mode)}</strong><small>${esc(date(run.created))}</small></div><div><span class="tag ${run.status === 'completed' ? 'green' : ['failed','interrupted'].includes(run.status) ? 'red' : 'neutral'}">${esc(statusNames[run.status] || run.status)}</span>${icon('arrow')}</div></button>`).join('') : '<div class="simple-empty">这个病例还没有分析记录。<br>点击「开始分析」生成第一份报告。</div>';
}

function renderProgress(run) {
  const events = run.events || [];
  const status = run.status;
  $('progress-panel').classList.remove('hidden');
  $('progress-panel').classList.toggle('failed', ['failed', 'interrupted'].includes(status));
  $('run-status').textContent = statusNames[status] || status;
  $('run-status').className = 'tag ' + (status === 'completed' ? 'green' : ['failed', 'interrupted'].includes(status) ? 'red' : 'neutral');
  $('progress-indicator').classList.toggle('hidden', ['completed', 'failed', 'interrupted'].includes(status));
  const stages = run.mode === 'direct' ? [
    ['prepare', '整理病例'],
    ['plan', '直接推理'],
    ['diagnose', '生成报告']
  ] : [
    ['prepare', '整理病例'],
    ['observe', '影像分析'],
    ['retrieve', '检索资料'],
    ['diagnose', '生成报告'],
    ['verify', '证据复核']
  ];
  const buckets = {
    prepare: 0,
    facts: 0,
    observe: 1,
    radiology: 1,
    plan: 1,
    tools: 1,
    retrieve: 2,
    contrast: 3,
    diagnose: 3,
    verify: 4,
    repair: 4,
    update: 4,
    complete: 4
  };
  const last = events.at(-1)?.stage;
  const current = run.mode === 'direct' ? last === 'complete' || last === 'diagnose' || last === 'update' ? 2 : last === 'plan' ? 1 : 0 : buckets[last] ?? 0;
  $('progress-steps').innerHTML = stages.map(([, label], i) => `<span class="progress-step ${status === 'completed' || i < current ? 'done' : i === current && status === 'running' ? 'current' : ''}">${label}</span>`).join('');
  const timeline = $('timeline');
  const wasBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 35;
  timeline.innerHTML = events.length ? events.map(e => `<div class="timeline-item"><span class="timeline-time">${esc(new Date(e.created).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',second:'2-digit'}))}</span><span class="timeline-stage">${esc(stageNames[e.stage] || e.stage)}</span><span>${esc(e.message)}</span></div>`).join('') : '<p class="subtle-text">任务已提交，等待远程服务开始处理。</p>';
  $('event-count').textContent = `${events.length} 条`;
  if (wasBottom) timeline.scrollTop = timeline.scrollHeight;
}
async function pollRun() {
  if (!state.activeRun || state.polling) return;
  state.polling = true;
  const id = state.activeRun;
  let retry = 1500;
  try {
    const run = await api('/api/runs/' + encodeURIComponent(id));
    if (state.activeRun !== id) return;
    renderProgress(run);
    if (['completed', 'failed', 'interrupted'].includes(run.status)) {
      state.activeRun = null;
      storage('chestActiveRun', null);
      if (run.result) {
        renderRun(run);
        message('分析已完成，可查看报告、证据和历史记录。');
      } else {
        $('report-content').innerHTML = `<div class="simple-empty"><span class="tag red">${esc(statusNames[run.status])}</span><p style="margin-top:15px">${esc(run.error || '任务未能完成，可重新提交。')}</p></div>`;
        message(run.error || '任务未能完成，可重新提交。', true);
      }
      if (state.selected) {
        state.history = await api('/api/cases/' + encodeURIComponent(state.selected.id) + '/runs');
        renderHistory();
      }
      renderCases();
      await health();
    }
  } catch (e) {
    message('进度连接暂时中断，正在重试。' + e.message, true);
    retry = 5000;
  } finally {
    state.polling = false;
    updateControls();
    if (state.activeRun) state.poll = setTimeout(pollRun, retry);
  }
}
async function startAnalysis() {
  if (!state.selected || state.activeRun || state.submitting || !state.health?.model_ready) return;
  const question = $('question').value.trim();
  if (!question) {
    message('请填写本次分析的问题。', true);
    return;
  }
  state.submitting = true;
  updateControls();
  message();
  try {
    const row = await api('/api/cases/' + encodeURIComponent(state.selected.id) + '/runs', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question,
        mode: state.mode
      })
    });
    state.activeRun = row.run_id;
    storage('chestActiveRun', JSON.stringify({
      run: row.run_id,
      case: state.selected.id
    }));
    clearReport();
    tab('report');
    $('report-content').innerHTML = `<div class="loading-text pending-report"><span class="spinner"></span>正在生成本次报告，分析进度会持续更新。</div>`;
    $('progress-panel').classList.remove('hidden');
    $('progress-details').open = true;
    message('任务已提交，在远程机器执行。');
    await pollRun();
  } catch (e) {
    message(e.message, true);
  } finally {
    state.submitting = false;
    updateControls();
  }
}
async function saveCase(event) {
  event.preventDefault();
  if (state.submitting || state.activeRun) return;
  const title = $('case-title').value.trim(),
    context = $('case-context-input').value.trim();
  if (!title || !context) {
    message('请填写病例标题和病史资料。', true);
    return;
  }
  state.submitting = true;
  updateControls();
  try {
    const body = new FormData();
    body.set('title', title);
    body.set('context', context);
    if (state.file) body.set('image', state.file);
    const c = await api('/api/cases', {
      method: 'POST',
      body
    });
    state.cases.unshift(c);
    await selectCase(c.id, true);
    message('病例已保存，可以开始分析。');
    toast('病例已保存');
  } catch (e) {
    message(e.message, true);
  } finally {
    state.submitting = false;
    updateControls();
  }
}
async function saveSupplement() {
  if (!state.selected || state.activeRun || state.submitting) return;
  const text = $('supplement').value.trim();
  if (!text) {
    message('请先填写新增资料。', true);
    return;
  }
  state.submitting = true;
  updateControls();
  try {
    const c = await api('/api/cases/' + encodeURIComponent(state.selected.id) + '/supplement', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text
      })
    });
    state.selected = c;
    state.cases = state.cases.map(row => row.id === c.id ? c : row);
    $('case-context').textContent = c.context;
    $('supplement').value = '';
    $('supplement-section').open = false;
    message('新增资料已保存。当前报告仍来自上一次分析，请重新分析以更新判断。');
    toast('补充资料已保存');
  } catch (e) {
    message(e.message, true);
  } finally {
    state.submitting = false;
    updateControls();
  }
}
async function copy(text, success) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const input = document.createElement('textarea');
      input.value = text;
      input.style.cssText = 'position:fixed;left:-9999px;top:0';
      (document.querySelector('dialog[open]') || document.body).append(input);
      input.focus();
      input.select();
      const ok = document.execCommand('copy');
      input.remove();
      if (!ok) throw Error('无法复制');
    }
    toast(success);
  } catch (_) {
    toast('复制未成功，请手动选择文字复制。');
  }
}

function renderConnection() {
  const h = state.health || {};
  $('access-url').value = location.origin;
  $('connection-details').innerHTML = `<div>连接状态：${state.connected ? h.inference_paused ? '推理暂停，历史报告可查看' : h.model_ready ? '服务已就绪' : '等待模型就绪' : '连接中断'}</div><div>当前模型：${esc(h.model_name || '—')}</div><div>模型加载：${h.model_loaded ? '已加载' : '首次分析时加载'}</div><div>资料库：${Number(h.knowledge_documents || 0)} 份来源文档</div>`;
}
async function loadEvaluation() {
  try {
    if (!state.evaluation) state.evaluation = await api('/static/evaluation.json');
    const d = state.evaluation,
      t = d.test,
      q = d.report_quality,
      c = d.development;
    $('evaluation-content').innerHTML = `<div class="eval-cards">
      <section class="panel eval-stat"><span>独立公开测试患者</span><h2>${d.test_patients} <small style="font-size:12px;font-weight:400">例</small></h2><p>按患者隔离，测试成绩不参与策略选择</p></section>
      <section class="panel eval-stat"><span>报告中的选择题结论</span><h2>${Math.round(t.accuracy*100)}%</h2><p>${t.correct}/${t.tasks} 题，失败任务计入分母</p></section>
      <section class="panel eval-stat"><span>完整报告生成</span><h2>${t.completed}/${t.tasks}</h2><p>模型：${esc(d.model)}</p></section></div>
      <section class="panel eval-panel"><h2>Agent 完整流程评测</h2><p>结合原始病史与胸片，执行图像工具、医学检索、临床证据对照、报告生成及证据复核。</p><div class="table-scroll"><table class="eval-table"><thead><tr><th>指标</th><th>结果</th></tr></thead><tbody>
      <tr><td>报告耗时中位数</td><td>${q.median_report_seconds} 秒</td></tr>
      <tr><td>GPU峰值保留显存</td><td>${q.gpu_peak_reserved_gib} GiB</td></tr>
      <tr><td>保留可用临床证据对照</td><td>${q.usable_contrasts}/${t.tasks}</td></tr>
      <tr><td>仍需复核的报告</td><td>${q.needs_review_reports}/${t.tasks}</td></tr>
      <tr><td>保留原文线索</td><td>${q.literal_clues_retained} 条</td></tr>
      </tbody></table></div><div class="eval-detail">待复核标记包含未完成的模型复核或证据对照、引用不匹配等问题，不能直接等同于临床错误数。</div>
      <div class="eval-footer"><span>快照导出：${esc(date(d.snapshot_exported_at))}</span><a href="/static/evaluation.json" download="agent-evaluation.json">下载评测汇总 ↓</a></div></section>
      <div class="eval-notes"><section class="panel eval-note"><h3>分数的含义</h3><p>这些分数衡量公开问题在完整报告中的选项结论，不代表临床诊断准确率。公开数据预训练污染情况未知，尚未经临床专家评分；引用检查通过也不能证明医学正确。</p></section>
      <section class="panel eval-note"><h3>开发验证与局限</h3><p>${c.inputs}组已知开发输入来自${c.patients}位患者，具体候选诊断名称符合参考结局的为${c.specific_matches}位，疾病大类为${c.broad_matches}位。这些是开发验证，不能作为独立准确率。结核仍误判，张力性气胸仍只识别到大类。</p></section></div>`;
  } catch (e) {
    $('evaluation-content').innerHTML = `<div class="panel simple-empty">评测记录读取失败：${esc(e.message)}<br><button class="text-button" id="retry-evaluation">重新读取</button></div>`;
  }
}
// Delegate only escaped identifiers; historical records never enter a new inference input.
document.addEventListener('click', async e => {
  const ref = e.target.closest('[data-ref]');
  if (ref) {
    tab('evidence');
    const target = $('evidence-' + ref.dataset.ref);
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
      target.classList.add('highlight');
      setTimeout(() => target.classList.remove('highlight'), 2500);
    } else toast('当前记录未找到这条证据。');
    return;
  }
  const c = e.target.closest('[data-case]');
  if (c) {
    await selectCase(c.dataset.case);
    sidebar(false);
    return;
  }
  const runButton = e.target.closest('[data-run]');
  if (runButton) {
    const token = state.selecting;
    try {
      const run = await api('/api/runs/' + encodeURIComponent(runButton.dataset.run));
      if (token !== state.selecting) return;
      if (run.result) {
        renderRun(run);
        tab('report');
      } else if (['queued', 'running'].includes(run.status)) {
        if (state.activeRun && state.activeRun !== run.id) return toast('当前有另一项分析进行中。');
        state.activeRun = run.id;
        storage('chestActiveRun', JSON.stringify({
          run: run.id,
          case: state.selected.id
        }));
        clearTimeout(state.poll);
        tab('report');
        await pollRun();
      } else {
        renderProgress(run);
        message(run.error || '这次分析没有完成，请重新提交。', true);
        tab('report');
      }
    } catch (error) {
      message(error.message, true);
    }
    return;
  }
  if (e.target.closest('#copy-conclusion') && state.reportRun) {
    const r = state.reportRun.result.report;
    await copy(`候选诊断：${r.most_likely.name}\n证据充分程度：${r.assessment}\n支持证据：\n${r.most_likely.support.map(c => c.text).join('\n')}\n\n研究用途，未经临床验证。`, '候选诊断已复制');
  }
  if (e.target.closest('#retry-evaluation')) loadEvaluation();
});
$('case-search').addEventListener('input', renderCases);
$('case-form').addEventListener('submit', saveCase);
$('save-supplement').addEventListener('click', saveSupplement);
$('analyze').addEventListener('click', startAnalysis);
for (const id of ['new-case', 'top-new-case']) $(id).addEventListener('click', () => selectCase(null));
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => view(b.dataset.view)));
document.querySelectorAll('[data-tab]').forEach(b => {
  b.addEventListener('click', () => tab(b.dataset.tab));
  b.addEventListener('keydown', e => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
      e.preventDefault();
      const tabs = [...document.querySelectorAll('[data-tab]')];
      const index = tabs.indexOf(b);
      const next = e.key === 'Home' ? tabs[0] : e.key === 'End' ? tabs.at(-1) : tabs[(index + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      tab(next.dataset.tab);
      next.focus();
    }
  });
});
document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('menu-toggle').addEventListener('click', () => sidebar(!$('sidebar').classList.contains('open')));
$('sidebar-scrim').addEventListener('click', () => sidebar(false));
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') sidebar(false);
});
$('dropzone').addEventListener('click', e => {
  if (e.target !== $('image-upload')) $('image-upload').click();
});
$('dropzone').addEventListener('keydown', e => {
  if (e.target === $('dropzone') && ['Enter', ' '].includes(e.key)) {
    e.preventDefault();
    $('image-upload').click();
  }
});
$('image-upload').addEventListener('change', e => chooseFile(e.target.files[0]));
$('clear-upload').addEventListener('click', resetUpload);
for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, e => {
  e.preventDefault();
  $('dropzone').classList.add('dragging');
});
for (const type of ['dragleave', 'drop']) $('dropzone').addEventListener(type, e => {
  e.preventDefault();
  $('dropzone').classList.remove('dragging');
  if (type === 'drop') chooseFile(e.dataTransfer.files[0]);
});
$('image-open').addEventListener('click', () => {
  $('full-image').src = $('preview').src;
  $('image-dialog').showModal();
});
$('connection-button').addEventListener('click', () => {
  renderConnection();
  $('connection-dialog').showModal();
});
$('copy-url').addEventListener('click', () => copy(location.origin, '访问地址已复制'));
document.querySelectorAll('[data-close-dialog]').forEach(b => b.addEventListener('click', () => $(b.dataset.closeDialog).close()));
document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click', e => {
  if (e.target === dialog) {
    const r = dialog.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
  }
}));
async function init() {
  setMode('verified');
  try {
    const [, cases] = await Promise.all([health(), api('/api/cases')]);
    state.cases = cases;
    let saved = null;
    try {
      saved = JSON.parse(storage('chestActiveRun') || 'null');
    } catch (_) {
      storage('chestActiveRun', null);
    }
    const initial = saved?.case || new URL(location.href).searchParams.get('case') || storage('chestSelectedCase');
    await selectCase(initial || null, true);
    if (saved?.run && state.selected?.id === saved.case) {
      state.activeRun = saved.run;
      clearReport();
      $('report-content').innerHTML = '<div class="loading-text pending-report"><span class="spinner"></span>已恢复任务，正在读取本次分析进度。</div>';
      clearTimeout(state.poll);
      await pollRun();
    } else if (saved) storage('chestActiveRun', null);
  } catch (e) {
    message(e.message, true);
    $('case-list').innerHTML = '<div class="sidebar-empty">病例读取失败，请检查服务连接后刷新。</div>';
  }
  updateControls();
}
init();
setInterval(health, 15000);
