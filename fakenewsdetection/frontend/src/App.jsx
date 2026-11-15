import { useEffect, useMemo, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const samples = [
  { title: "NASA announces a new space research mission", category: "Science", text: "NASA announced a new scientific mission designed to study distant celestial objects. Researchers will analyse the observations collected by the spacecraft and publish findings after the data has been reviewed by the mission team." },
  { title: "Free laptop scheme requires no application", category: "Technology", text: "A viral message claims that every resident will receive a free laptop immediately. It asks readers to submit personal information through an unfamiliar website and forward the message to ten contacts to claim the benefit." },
  { title: "National team wins international tournament", category: "Sports", text: "The national team secured the tournament title after a closely contested final. Players credited their preparation and teamwork, while the coaching staff said a detailed review of the campaign would follow." },
];

function downloadCSV(rows) {
  const columns = ["id", "title", "category", "prediction", "confidence", "created_at"];
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [columns.join(","), ...rows.map(row => columns.map(key => quote(row[key])).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = "news-prediction-history.csv"; link.click();
  URL.revokeObjectURL(url);
}

function App() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [category, setCategory] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [apiOnline, setApiOnline] = useState(false);

  const loadData = async () => {
    try {
      const [h, m] = await Promise.all([fetch(`${API}/history`), fetch(`${API}/metrics`)]);
      if (!h.ok) throw new Error("History unavailable");
      const hd = await h.json(); setHistory(hd.history || []);
      if (m.ok) setMetrics(await m.json());
      setApiOnline(true);
    } catch { setApiOnline(false); }
  };
  useEffect(() => { loadData(); }, []);

  const fakeCount = history.filter(x => String(x.prediction).toUpperCase() === "FAKE").length;
  const realCount = history.filter(x => String(x.prediction).toUpperCase() === "REAL").length;
  const filteredHistory = useMemo(() => history.filter(item => {
    const matchesFilter = filter === "ALL" || String(item.prediction).toUpperCase() === filter;
    const haystack = `${item.title} ${item.category} ${item.prediction}`.toLowerCase();
    return matchesFilter && haystack.includes(query.toLowerCase());
  }), [history, filter, query]);
  const chartData = [{ name: "Fake", value: fakeCount }, { name: "Real", value: realCount }];

  const checkNews = async (event) => {
    event.preventDefault();
    if (!title.trim() || !text.trim()) { setError("Please enter both a news title and the article content."); return; }
    setLoading(true); setError(""); setNotice(""); setResult(null);
    try {
      const response = await fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: title.trim(), text: text.trim(), category }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed. Please try again.");
      setResult(data.result); setNotice("Analysis completed and saved to your prediction history.");
      await loadData();
    } catch (err) { setError(err.message.includes("Failed to fetch") ? "Cannot connect to the backend. Start the FastAPI server and try again." : err.message); }
    finally { setLoading(false); }
  };
  const useSample = () => { const sample = samples[Math.floor(Math.random() * samples.length)]; setTitle(sample.title); setText(sample.text); setCategory(sample.category); setResult(null); setError(""); setNotice("Sample article loaded. Click Analyse News to test the model."); };
  const clearForm = () => { setTitle(""); setText(""); setCategory(""); setResult(null); setError(""); setNotice(""); };
  const clearHistory = async () => {
    if (!window.confirm("Delete all saved prediction history? This cannot be undone.")) return;
    try { const response = await fetch(`${API}/history`, { method: "DELETE" }); if (!response.ok) throw new Error(); await loadData(); setNotice("Prediction history cleared."); }
    catch { setError("Could not clear history. Check that the backend is running."); }
  };

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#home"><span className="brand-mark">N</span><span>News<span className="brand-accent">Verify</span><small>INTELLIGENCE STUDIO</small></span></a>
      <nav>{[["Home","home"],["Dashboard","dashboard"],["Check News","check-news"],["Analytics","analytics"],["History","history"],["About","about"],["Disclaimer","disclaimer"]].map(([label,id]) => <a key={id} href={`#${id}`}>{label}</a>)}</nav>
      <span className={`status ${apiOnline ? "online" : "offline"}`}><i />{apiOnline ? "API connected" : "API offline"}</span>
    </header>

    <main>
      <section id="home" className="hero section-anchor"><div className="hero-copy"><span className="eyebrow">AI-POWERED NEWS INTELLIGENCE</span><h1>Separate the facts<br/><em>from the noise.</em></h1><p>Analyse news articles with Natural Language Processing and Machine Learning. Get a model-generated classification with probability insights in seconds.</p><div className="hero-actions"><a className="button primary" href="#check-news">Analyse an article <span>↗</span></a><a className="text-link" href="#about">Explore the project ↓</a></div><div className="hero-note"><span>✦</span> A decision-support tool, not a replacement for fact-checking.</div></div>
      <div className="hero-art"><div className="orb orb-one"/><div className="orb orb-two"/><div className="news-visual"><div className="visual-top"><span>NEWS INTELLIGENCE</span><b>✦ LIVE</b></div><div className="visual-icon">◎</div><h3>Read critically.<br/>Verify thoughtfully.</h3><div className="visual-lines"><i/><i/><i/></div><div className="visual-bottom"><span>AI / NLP / ML</span><span>01 — 03</span></div></div><div className="floating-chip chip-a">✦ Text analysis</div><div className="floating-chip chip-b">◎ Probability insights</div></div></section>

      <section id="dashboard" className="section-anchor"><div className="section-heading"><div><span className="eyebrow">OVERVIEW</span><h2>Your analysis at a glance</h2></div><span className="muted">Live totals from saved predictions</span></div>
        <div className="stats-grid"><article className="stat-card stat-purple"><span className="stat-icon">▤</span><span className="stat-label">TOTAL ANALYSES</span><strong>{history.length}</strong><small>Articles processed</small></article><article className="stat-card stat-pink"><span className="stat-icon">⚑</span><span className="stat-label">CLASSIFIED FAKE</span><strong>{fakeCount}</strong><small>Model predictions</small></article><article className="stat-card stat-teal"><span className="stat-icon">✓</span><span className="stat-label">CLASSIFIED REAL</span><strong>{realCount}</strong><small>Model predictions</small></article><article className="stat-card stat-gold"><span className="stat-icon">◷</span><span className="stat-label">MODEL STATUS</span><strong className="status-word">{apiOnline ? "Ready" : "Offline"}</strong><small>{apiOnline ? "Backend reachable" : "Start backend to analyse"}</small></article></div>
      </section>

      <section id="check-news" className="section-anchor analysis-layout"><div className="section-heading"><div><span className="eyebrow">CONTENT ANALYSIS</span><h2>Check a news article</h2><p className="muted">Enter the article details below to receive a classification.</p></div><button className="button subtle" type="button" onClick={useSample}>✦ Try sample news</button></div>
        <div className="form-card"><form onSubmit={checkNews}><div className="form-grid"><label>News headline<input value={title} onChange={e=>setTitle(e.target.value)} placeholder="Enter the article headline" maxLength={300}/></label><label>News category<select value={category} onChange={e=>setCategory(e.target.value)}><option value="">Select a category (optional)</option>{["Politics","Business","Technology","Science","Health","Sports","Entertainment","World","Other"].map(x=><option key={x}>{x}</option>)}</select></label></div><label>Article content<textarea value={text} onChange={e=>setText(e.target.value)} placeholder="Paste or type the complete news article here..." rows="8"/></label><div className="form-footer"><span>{text.length.toLocaleString()} characters</span><div><button className="button subtle" type="button" onClick={clearForm}>Clear fields</button><button className="button primary" type="submit" disabled={loading}>{loading ? <><span className="spinner"/> Analysing...</> : "Analyse news →"}</button></div></div></form>
          {error && <div className="alert error">{error}</div>}{notice && <div className="alert success">{notice}</div>}
        </div>
        {result && <article className={`result-panel ${String(result.prediction).toUpperCase()==="FAKE"?"is-fake":"is-real"}`}><div className="result-head"><div><span className="eyebrow">MODEL OUTPUT</span><h2>Prediction result</h2></div><span className="result-badge">{String(result.prediction).toUpperCase()==="FAKE"?"⚑":"✓"} {result.prediction}</span></div><p className="result-disclaimer">This is an automated classification based on learned text patterns. It does not independently verify the article's claims.</p><div className="probability-grid"><div><span>Model confidence</span><strong>{Number(result.confidence).toFixed(2)}%</strong><div className="progress"><i style={{width:`${Math.min(100,Math.max(0,result.confidence))}%`}}/></div></div><div><span>Fake probability</span><strong>{Number(result.fake_probability).toFixed(2)}%</strong><div className="progress fake-progress"><i style={{width:`${Math.min(100,Math.max(0,result.fake_probability))}%`}}/></div></div><div><span>Real probability</span><strong>{Number(result.real_probability).toFixed(2)}%</strong><div className="progress real-progress"><i style={{width:`${Math.min(100,Math.max(0,result.real_probability))}%`}}/></div></div></div></article>}
      </section>

      <section id="analytics" className="section-anchor"><div className="section-heading"><div><span className="eyebrow">INSIGHTS</span><h2>Prediction analytics</h2></div><span className="muted">Based on your saved analysis history</span></div><div className="analytics-grid"><article className="panel chart-panel"><h3>Classification distribution</h3>{history.length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height={280}><PieChart><Pie data={chartData} dataKey="value" nameKey="name" innerRadius={68} outerRadius={100} paddingAngle={4}>{chartData.map((entry,index)=><Cell key={entry.name} fill={index===0?"#f06b91":"#20b99a"}/>)}</Pie><Tooltip/><Legend/></PieChart></ResponsiveContainer></div> : <div className="empty-chart">Your chart will appear after the first prediction.</div>}</article><article className="panel metrics-panel"><h3>Model performance</h3><p className="muted small">Evaluation metrics from the model comparison report included with this project.</p>{metrics?.models?.length ? metrics.models.map(model=><div className="model-row" key={model.model}><div className="model-name"><b>{model.model}</b><span>{(model.accuracy*100).toFixed(2)}% accuracy</span></div><div className="metric-bars">{[["Precision",model.precision],["Recall",model.recall],["F1-score",model.f1_score]].map(([name,value])=><div className="metric-line" key={name}><span>{name}</span><div className="progress"><i style={{width:`${Math.max(0,Math.min(100,value*100))}%`}}/></div><b>{(value*100).toFixed(1)}%</b></div>)}</div></div>) : <p className="empty-chart">Performance report is not available. See backend/model_comparison_results.csv.</p>}<p className="metric-warning">Metrics describe the supplied test evaluation and may vary with dataset quality. Review the dataset and retrain before making reliability claims.</p></article></div></section>

      <section id="history" className="section-anchor"><div className="section-heading"><div><span className="eyebrow">YOUR RECORDS</span><h2>Prediction history</h2><p className="muted">Search, filter and export previously analysed articles.</p></div><div className="history-actions"><button className="button subtle" onClick={()=>downloadCSV(filteredHistory)} disabled={!filteredHistory.length}>↓ Export CSV</button><button className="button danger-outline" onClick={clearHistory} disabled={!history.length}>Clear history</button></div></div><div className="panel history-panel"><div className="history-toolbar"><input aria-label="Search history" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search by headline, category or result..."/><select value={filter} onChange={e=>setFilter(e.target.value)} aria-label="Filter prediction"><option value="ALL">All predictions</option><option value="FAKE">Fake only</option><option value="REAL">Real only</option></select></div>{filteredHistory.length ? <div className="table-scroll"><table><thead><tr><th>Article</th><th>Category</th><th>Prediction</th><th>Confidence</th><th>Date</th></tr></thead><tbody>{filteredHistory.map(item=><tr key={item.id}><td className="article-title">{item.title}</td><td>{item.category||"—"}</td><td><span className={`pill ${String(item.prediction).toLowerCase()}`}>{item.prediction}</span></td><td>{Number(item.confidence).toFixed(2)}%</td><td>{item.created_at}</td></tr>)}</tbody></table></div> : <div className="empty-chart">No matching predictions found. Analyse an article to get started.</div>}</div></section>

      <section id="about" className="section-anchor about-section"><div className="section-heading"><div><span className="eyebrow">THE PROJECT</span><h2>About NewsVerify AI</h2></div></div><div className="about-grid"><article className="panel about-copy"><h3>Intelligent Fake News Detection and Analysis System Using NLP and Machine Learning</h3><p>This academic project classifies submitted news text as FAKE or REAL using a trained machine-learning model. It combines the headline, article body and optional category, then transforms the text into numerical features for classification.</p><p>The application is designed to demonstrate an end-to-end workflow: text input, preprocessing, TF-IDF feature extraction, model prediction, probability display and database-backed history.</p></article><article className="panel tech-panel"><h3>Technology stack</h3><div className="tech-tags">{["React","Vite","Python","FastAPI","NLP","TF-IDF","Logistic Regression","Scikit-learn","SQLite","Recharts"].map(t=><span key={t}>{t}</span>)}</div><h3 className="method-title">Methodology</h3><ol><li>Collect and prepare labelled news data.</li><li>Clean and preprocess text.</li><li>Convert text into TF-IDF features.</li><li>Train and evaluate a classification model.</li><li>Predict, store and visualise results.</li></ol></article></div></section>

      <section id="disclaimer" className="section-anchor disclaimer"><span className="disclaimer-icon">ⓘ</span><div><h3>Important disclaimer</h3><p>This system is an educational machine-learning demonstration, not a professional fact-checking service. A FAKE or REAL label is a model prediction, not proof. Confidence is not the same as factual certainty. Always verify important claims using multiple reliable, independent sources before sharing or acting on them.</p></div></section>
    </main>
    <footer><a className="brand footer-brand" href="#home"><span className="brand-mark">N</span><span>News<span className="brand-accent">Verify</span></span></a><span>Intelligent Fake News Detection & Analysis · MCA Project</span><a href="#home">Back to top ↑</a></footer>
  </div>;
}
export default App;
