/* MicroRTS LLM Competition - Leaderboard App */

(function () {
    'use strict';

    var DATA_URL = 'data/tournament_results.json';
    var ANCHORS = ['RandomBiasedAI', 'HeavyRush', 'LightRush', 'WorkerRush', 'Tiamat', 'CoacAI'];
    var GRADE_ORDER = { 'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5 };

    // Store raw data for filtering/sorting
    var rawData = null;
    var sortState = {}; // tableId -> { key, dir }

    // Tab switching
    document.querySelectorAll('.tab').forEach(function (tab) {
        tab.addEventListener('click', function (e) {
            e.preventDefault();
            var target = this.getAttribute('data-tab');
            if (!target) return;

            document.querySelectorAll('.tab').forEach(function (t) {
                t.classList.remove('active');
            });
            this.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(function (c) {
                c.classList.remove('active');
            });
            var el = document.getElementById(target);
            if (el) el.classList.add('active');
        });
    });

    // Grade badge HTML
    function gradeBadge(grade) {
        var cls = 'grade-f';
        if (grade === 'A+') cls = 'grade-aplus';
        else if (grade === 'A') cls = 'grade-a';
        else if (grade === 'B') cls = 'grade-b';
        else if (grade === 'C') cls = 'grade-c';
        else if (grade === 'D') cls = 'grade-d';
        return '<span class="grade ' + cls + '">' + escapeHtml(grade) + '</span>';
    }

    function escapeHtml(str) {
        if (str == null) return '';
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // Format opponent cell
    function opponentCell(opponents, anchorName) {
        var data = opponents ? opponents[anchorName] : null;
        if (!data) return '<td class="result-skip">--</td>';

        var w = data.wins || 0;
        var d = data.draws || 0;
        var l = data.losses || 0;
        var pts = data.weighted_points != null ? data.weighted_points : '?';

        var cls = 'result-skip';
        if (w > 0) cls = 'result-win';
        else if (l > 0) cls = 'result-loss';
        else if (d > 0) cls = 'result-draw';

        return '<td class="' + cls + '">' + w + 'W/' + d + 'D/' + l + 'L<br><small>' + pts + ' pts</small></td>';
    }

    // Format date
    function formatDate(dateStr) {
        if (!dateStr) return '--';
        try {
            var d = new Date(dateStr);
            return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        } catch (e) {
            return dateStr.substring(0, 10);
        }
    }

    // Relative time (e.g., "3 days ago")
    function timeAgo(dateStr) {
        if (!dateStr) return '';
        try {
            var d = new Date(dateStr);
            var now = new Date();
            var diffMs = now - d;
            var diffMins = Math.floor(diffMs / 60000);
            var diffHours = Math.floor(diffMs / 3600000);
            var diffDays = Math.floor(diffMs / 86400000);

            if (diffMins < 1) return 'just now';
            if (diffMins < 60) return diffMins + 'm ago';
            if (diffHours < 24) return diffHours + 'h ago';
            if (diffDays === 1) return '1 day ago';
            if (diffDays < 30) return diffDays + ' days ago';
            if (diffDays < 365) return Math.floor(diffDays / 30) + ' months ago';
            return Math.floor(diffDays / 365) + ' years ago';
        } catch (e) {
            return '';
        }
    }

    // Format date with relative time for leaderboard
    function formatDateWithAge(dateStr) {
        if (!dateStr) return '--';
        var formatted = formatDate(dateStr);
        var relative = timeAgo(dateStr);
        if (relative) {
            return formatted + '<br><small class="time-ago">' + relative + '</small>';
        }
        return formatted;
    }

    // Extract YYYY-MM-DD from a date string for comparison
    function toDateOnly(dateStr) {
        if (!dateStr) return '';
        try {
            return new Date(dateStr).toISOString().substring(0, 10);
        } catch (e) {
            return '';
        }
    }

    // --- Sorting ---

    function comparator(type) {
        return function (a, b) {
            if (type === 'number') {
                return (parseFloat(a) || 0) - (parseFloat(b) || 0);
            }
            if (type === 'date') {
                var da = a ? new Date(a).getTime() : 0;
                var db = b ? new Date(b).getTime() : 0;
                return da - db;
            }
            if (type === 'grade') {
                var ga = GRADE_ORDER[a] != null ? GRADE_ORDER[a] : 99;
                var gb = GRADE_ORDER[b] != null ? GRADE_ORDER[b] : 99;
                return ga - gb;
            }
            // string
            return String(a || '').localeCompare(String(b || ''));
        };
    }

    function sortEntries(entries, key, type, dir) {
        var cmp = comparator(type);
        var sorted = entries.slice();
        sorted.sort(function (a, b) {
            var va = a._sortData[key];
            var vb = b._sortData[key];
            var result = cmp(va, vb);
            return dir === 'asc' ? result : -result;
        });
        return sorted;
    }

    function updateSortIndicators(tableId, activeKey, dir) {
        var table = document.getElementById(tableId);
        if (!table) return;
        table.querySelectorAll('th[data-sort]').forEach(function (th) {
            th.classList.remove('sort-asc', 'sort-desc');
            if (th.getAttribute('data-sort') === activeKey) {
                th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
            }
        });
    }

    function attachSortHandlers(tableId, renderFn) {
        var table = document.getElementById(tableId);
        if (!table) return;
        table.querySelectorAll('th[data-sort]').forEach(function (th) {
            th.style.cursor = 'pointer';
            th.addEventListener('click', function () {
                var key = this.getAttribute('data-sort');
                var type = this.getAttribute('data-type') || 'string';
                var state = sortState[tableId] || {};
                var dir = (state.key === key && state.dir === 'desc') ? 'asc' : 'desc';
                sortState[tableId] = { key: key, type: type, dir: dir };
                updateSortIndicators(tableId, key, dir);
                renderFn();
            });
        });
    }

    // --- Filtering ---

    function getFilterValues(prefix) {
        var searchEl = document.getElementById(prefix + '-search');
        var fromEl = document.getElementById(prefix + '-date-from');
        var toEl = document.getElementById(prefix + '-date-to');
        return {
            search: searchEl ? searchEl.value.toLowerCase().trim() : '',
            dateFrom: fromEl ? fromEl.value : '',
            dateTo: toEl ? toEl.value : ''
        };
    }

    function matchesSearch(entry, searchStr, fields) {
        if (!searchStr) return true;
        for (var i = 0; i < fields.length; i++) {
            var val = entry._sortData[fields[i]];
            if (val && String(val).toLowerCase().indexOf(searchStr) !== -1) return true;
        }
        return false;
    }

    function matchesDateRange(dateStr, from, to) {
        if (!from && !to) return true;
        var d = toDateOnly(dateStr);
        if (!d) return !from && !to;
        if (from && d < from) return false;
        if (to && d > to) return false;
        return true;
    }

    function updateCount(id, shown, total) {
        var el = document.getElementById(id);
        if (!el) return;
        if (shown === total) {
            el.textContent = total + ' entries';
        } else {
            el.textContent = 'Showing ' + shown + ' of ' + total + ' entries';
        }
    }

    function attachFilterHandlers(prefix, renderFn) {
        var ids = [prefix + '-search', prefix + '-date-from', prefix + '-date-to'];
        ids.forEach(function (id) {
            var el = document.getElementById(id);
            if (el) el.addEventListener('input', renderFn);
        });
        var clearBtn = document.getElementById(prefix + '-date-clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                var fromEl = document.getElementById(prefix + '-date-from');
                var toEl = document.getElementById(prefix + '-date-to');
                if (fromEl) fromEl.value = '';
                if (toEl) toEl.value = '';
                renderFn();
            });
        }
    }

    // --- Render functions ---

    // Leaderboard
    var leaderboardEntries = [];

    function prepareLeaderboard(data) {
        leaderboardEntries = (data.leaderboard || []).map(function (entry, i) {
            return {
                raw: entry,
                _sortData: {
                    rank: i + 1,
                    name: entry.display_name || '',
                    score: entry.score,
                    grade: entry.grade || 'F',
                    date: entry.last_updated || entry.date || ''
                }
            };
        });
    }

    function renderLeaderboard() {
        var tbody = document.querySelector('#leaderboard-table tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        var filters = getFilterValues('leaderboard');
        var entries = leaderboardEntries.filter(function (e) {
            if (!matchesSearch(e, filters.search, ['name', 'grade'])) return false;
            if (!matchesDateRange(e._sortData.date, filters.dateFrom, filters.dateTo)) return false;
            return true;
        });

        var state = sortState['leaderboard-table'];
        if (state) {
            entries = sortEntries(entries, state.key, state.type, state.dir);
        }

        entries.forEach(function (e, i) {
            var entry = e.raw;
            var row = document.createElement('tr');
            var displayRank = state ? (i + 1) : e._sortData.rank;
            var html = '<td class="rank-cell">' + displayRank + '</td>';
            html += '<td>' + escapeHtml(entry.display_name) + '</td>';
            html += '<td class="score-cell">' + entry.score + '</td>';
            html += '<td>' + gradeBadge(entry.grade) + '</td>';
            html += '<td class="date-cell">' + formatDateWithAge(entry.last_updated || entry.date) + '</td>';

            ANCHORS.forEach(function (anchor) {
                html += opponentCell(entry.opponents, anchor);
            });

            row.innerHTML = html;
            tbody.appendChild(row);
        });

        updateCount('leaderboard-count', entries.length, leaderboardEntries.length);
    }

    // Benchmarks
    var benchmarkEntries = [];

    function prepareBenchmarks(data) {
        var entries = data.history || [];
        var seen = {};
        entries.forEach(function (entry) {
            var key = entry.display_name;
            if (!seen[key] || entry.score > seen[key].score) {
                seen[key] = entry;
            }
        });
        var best = Object.values(seen);
        best.sort(function (a, b) { return b.score - a.score; });

        benchmarkEntries = best.map(function (entry) {
            return {
                raw: entry,
                _sortData: {
                    name: entry.display_name || '',
                    score: entry.score,
                    grade: entry.grade || 'F',
                    source: entry.source || 'unknown',
                    date: entry.date || ''
                }
            };
        });
    }

    function renderBenchmarks() {
        var tbody = document.querySelector('#benchmarks-table tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        var filters = getFilterValues('benchmarks');
        var entries = benchmarkEntries.filter(function (e) {
            if (!matchesSearch(e, filters.search, ['name', 'grade', 'source'])) return false;
            if (!matchesDateRange(e._sortData.date, filters.dateFrom, filters.dateTo)) return false;
            return true;
        });

        var state = sortState['benchmarks-table'];
        if (state) {
            entries = sortEntries(entries, state.key, state.type, state.dir);
        }

        entries.forEach(function (e) {
            var entry = e.raw;
            var row = document.createElement('tr');
            var src = entry.source || 'unknown';
            row.innerHTML =
                '<td>' + escapeHtml(entry.display_name) + '</td>' +
                '<td class="score-cell">' + entry.score + '</td>' +
                '<td>' + gradeBadge(entry.grade) + '</td>' +
                '<td><span class="source-badge">' + escapeHtml(src) + '</span></td>' +
                '<td>' + formatDate(entry.date) + '</td>';
            tbody.appendChild(row);
        });

        updateCount('benchmarks-count', entries.length, benchmarkEntries.length);
    }

    // Head-to-head matrix
    function renderH2H(data) {
        var h2h = data.head_to_head || {};
        var players = Object.keys(h2h).sort();

        if (players.length === 0) {
            document.getElementById('h2h-empty').style.display = 'block';
            document.querySelector('#h2h .table-container').style.display = 'none';
            return;
        }

        var thead = document.querySelector('#h2h-table thead');
        var tbody = document.querySelector('#h2h-table tbody');

        var headerHtml = '<tr><th></th>';
        players.forEach(function (p) {
            headerHtml += '<th>' + escapeHtml(p.split(' (')[0]) + '</th>';
        });
        headerHtml += '</tr>';
        thead.innerHTML = headerHtml;

        tbody.innerHTML = '';
        players.forEach(function (p1) {
            var row = document.createElement('tr');
            var html = '<td>' + escapeHtml(p1) + '</td>';

            players.forEach(function (p2) {
                if (p1 === p2) {
                    html += '<td class="h2h-self">-</td>';
                } else {
                    var record = h2h[p1] && h2h[p1][p2];
                    if (record) {
                        var cls = 'result-skip';
                        if (record.wins > record.losses) cls = 'result-win';
                        else if (record.losses > record.wins) cls = 'result-loss';
                        else if (record.draws > 0) cls = 'result-draw';
                        html += '<td class="' + cls + '">' + record.wins + 'W/' + record.draws + 'D/' + record.losses + 'L</td>';
                    } else {
                        html += '<td class="result-skip">--</td>';
                    }
                }
            });

            row.innerHTML = html;
            tbody.appendChild(row);
        });
    }

    // History
    var historyEntries = [];

    function prepareHistory(data) {
        historyEntries = (data.history || []).map(function (entry) {
            return {
                raw: entry,
                _sortData: {
                    date: entry.date || '',
                    name: entry.display_name || '',
                    score: entry.score,
                    grade: entry.grade || 'F',
                    source: entry.source || 'unknown',
                    map: entry.map || ''
                }
            };
        });
    }

    function renderHistory() {
        var tbody = document.querySelector('#history-table tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        var filters = getFilterValues('history');
        var entries = historyEntries.filter(function (e) {
            if (!matchesSearch(e, filters.search, ['name', 'grade', 'source', 'map'])) return false;
            if (!matchesDateRange(e._sortData.date, filters.dateFrom, filters.dateTo)) return false;
            return true;
        });

        var state = sortState['history-table'];
        if (state) {
            entries = sortEntries(entries, state.key, state.type, state.dir);
        }

        entries.forEach(function (e) {
            var entry = e.raw;
            var row = document.createElement('tr');
            var src = entry.source || 'unknown';
            row.innerHTML =
                '<td>' + formatDate(entry.date) + '</td>' +
                '<td>' + escapeHtml(entry.display_name) + '</td>' +
                '<td class="score-cell">' + entry.score + '</td>' +
                '<td>' + gradeBadge(entry.grade) + '</td>' +
                '<td><span class="source-badge">' + escapeHtml(src) + '</span></td>' +
                '<td>' + escapeHtml(entry.map || '--') + '</td>';
            tbody.appendChild(row);
        });

        updateCount('history-count', entries.length, historyEntries.length);
    }

    // --- Init ---

    fetch(DATA_URL)
        .then(function (response) {
            if (!response.ok) throw new Error('Failed to load data: ' + response.status);
            return response.json();
        })
        .then(function (data) {
            rawData = data;

            // Prepare data
            prepareLeaderboard(data);
            prepareBenchmarks(data);
            prepareHistory(data);

            // Initial render
            renderLeaderboard();
            renderBenchmarks();
            renderH2H(data);
            renderHistory();

            // Attach sort handlers
            attachSortHandlers('leaderboard-table', renderLeaderboard);
            attachSortHandlers('benchmarks-table', renderBenchmarks);
            attachSortHandlers('history-table', renderHistory);

            // Attach filter handlers
            attachFilterHandlers('leaderboard', renderLeaderboard);
            attachFilterHandlers('benchmarks', renderBenchmarks);
            attachFilterHandlers('history', renderHistory);

            var genEl = document.getElementById('generated-at');
            if (genEl && data.generated) {
                genEl.textContent = 'Data updated: ' + formatDate(data.generated);
            }
        })
        .catch(function (err) {
            console.error('Error loading tournament data:', err);
            var main = document.querySelector('main');
            if (main) {
                main.innerHTML = '<p class="empty-msg">Failed to load tournament data. ' +
                    'If viewing locally, serve with a local HTTP server (e.g. <code>python3 -m http.server</code> in the docs/ directory).</p>';
            }
        });
})();
