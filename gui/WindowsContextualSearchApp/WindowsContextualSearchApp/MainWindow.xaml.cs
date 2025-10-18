using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;      //Hyperlink support
using WinForms = System.Windows.Forms;
using System.Windows.Navigation;

namespace WindowsContextualSearchApp
{
    public partial class MainWindow : Window
    {
        //HttpClient used for sending API requests
        private readonly HttpClient _http = new HttpClient();

        //Holds the list of file hits returned by the backend
        private readonly ObservableCollection<FileItem> _results = new ObservableCollection<FileItem>();

        public MainWindow()
        {
            InitializeComponent();

            //Make sure UI matches current engine mode on startup
            RefreshEngineUI();

            //Log that the app has started
            Log("App started.");
        }

        //Opens a folder browser so the user can pick their workspace
        private void BrowseFolder_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                using var dlg = new WinForms.FolderBrowserDialog();
                var r = dlg.ShowDialog();
                if (r == WinForms.DialogResult.OK)
                    DataFolderBox.Text = dlg.SelectedPath; //Save the picked path in the textbox
            }
            catch (Exception ex) { Log($"Folder pick error: {ex.Message}"); }
        }

        //Simple health check against the backend /health route
        private async void HealthCheck_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var url = ApiBaseBox.Text.TrimEnd('/') + "/health";
                var resp = await _http.GetAsync(url);
                resp.EnsureSuccessStatusCode();
                var json = await resp.Content.ReadAsStringAsync();
                Status($"API OK: {json}");
                Log($"Health: {json}");
            }
            catch (Exception ex)
            {
                Status("API health check failed");
                Log(ex.ToString());
            }
        }

        //When user changes the selected engine (LLM Agent / Router)
        private void EngineBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            RefreshEngineUI();
        }

        //Toggles visibility of the “Steps” box depending on engine mode
        private void RefreshEngineUI()
        {
            if (StepsWrap == null || EngineBox == null) return; //During InitializeComponent, controls aren’t ready yet

            var engine = (EngineBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Router";
            StepsWrap.Visibility = string.Equals(engine, "LLM Agent", StringComparison.OrdinalIgnoreCase)
                ? Visibility.Visible
                : Visibility.Collapsed;
        }

        //Triggered when user clicks “Search”
        private async void Search_Click(object sender, RoutedEventArgs e)
        {
            //Clear previous results and reset UI
            _results.Clear();
            AnswerText.Text = "";
            CitationsPanel.ItemsSource = null;

            //Basic validation for required fields
            var baseUrl = ApiBaseBox.Text.TrimEnd('/');
            if (string.IsNullOrWhiteSpace(baseUrl)) { Status("Set API base URL first."); return; }

            var q = PromptBox.Text;
            if (string.IsNullOrWhiteSpace(q)) { Status("Type a prompt first."); return; }

            //Router parameters
            int k = SafeInt(KBox.Text, 6);
            int ck = SafeInt(CKBox.Text, 8);
            int bk = SafeInt(BKBox.Text, 20);
            string before = string.IsNullOrWhiteSpace(BeforeBox.Text) ? null : BeforeBox.Text.Trim();

            //Agent parameter
            int steps = SafeInt(StepsBox.Text, 4);

            //Optional workspace path (picked folder)
            var workspace = string.IsNullOrWhiteSpace(DataFolderBox.Text) ? null : DataFolderBox.Text.Trim();

            //Current engine mode
            var engine = (EngineBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Router";

            Status("Searching…");

            try
            {
                HttpResponseMessage resp;

                //Decide which backend endpoint to hit depending on engine mode
                if (string.Equals(engine, "LLM Agent", StringComparison.OrdinalIgnoreCase))
                {
                    Log($"POST /agent q='{q}', k={k}, steps={steps}, before={before ?? "-"}, workspace={workspace ?? "-"}");
                    var payload = new
                    {
                        question = q,
                        k,
                        steps,
                        before,
                        workspace
                    };
                    resp = await _http.PostAsJsonAsync($"{baseUrl}/agent", payload);
                }
                else
                {
                    Log($"POST /route q='{q}', k={k}, ck={ck}, bk={bk}, before={before ?? "-"}, workspace={workspace ?? "-"}");
                    var payload = new
                    {
                        q,
                        k,
                        ck,
                        bk,
                        before,
                        workspace
                    };
                    resp = await _http.PostAsJsonAsync($"{baseUrl}/route", payload);
                }

                resp.EnsureSuccessStatusCode();

                //Parse JSON response from backend
                using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
                var root = doc.RootElement;

                var mode = root.GetProperty("mode").GetString();

                //If backend returned file hits instead of an answer
                if (string.Equals(mode, "files", StringComparison.OrdinalIgnoreCase))
                {
                    if (root.TryGetProperty("files", out var filesEl) && filesEl.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var f in filesEl.EnumerateArray())
                        {
                            var path = f.TryGetProperty("path", out var pth) ? pth.GetString()
                                     : f.TryGetProperty("source", out var src) ? src.GetString()
                                     : "";

                            var page = f.TryGetProperty("top_page", out var tp) ? tp.GetInt32()
                                     : f.TryGetProperty("page", out var pg) ? pg.GetInt32()
                                     : 0;

                            var preview = f.TryGetProperty("preview", out var pv) ? pv.GetString()
                                         : f.TryGetProperty("snippet", out var sn) ? sn.GetString()
                                         : "";

                            _results.Add(new FileItem { Path = path ?? "", Page = page, Snippet = preview ?? "" });
                        }
                        Status($"Found {_results.Count} file hits.");
                        Log($"Files mode: {_results.Count} items.");
                    }
                    else
                    {
                        Status("No files returned.");
                    }
                }
                else
                {
                    //Otherwise, backend sent a direct text answer
                    var answer = root.TryGetProperty("answer", out var a) ? a.GetString() : "(no answer)";
                    AnswerText.Text = answer ?? "";

                    //Parse any citations to display clickable links
                    var citations = new ObservableCollection<Citation>();
                    if (root.TryGetProperty("citations", out var citsEl) && citsEl.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var c in citsEl.EnumerateArray())
                        {
                            citations.Add(new Citation
                            {
                                Path = c.GetProperty("path").GetString() ?? c.GetProperty("source").GetString(),
                                Page = c.TryGetProperty("page", out var p) ? p.GetInt32() : 0,
                                ChunkId = c.TryGetProperty("chunk_id", out var id) ? id.GetString() : null
                            });
                        }
                    }
                    CitationsPanel.ItemsSource = citations;

                    Status("Answer ready.");
                    Log($"Answer mode: {answer?.Length} chars, {CitationsPanel.Items.Count} citations.");
                }
            }
            catch (Exception ex)
            {
                Status("Request failed.");
                Log(ex.ToString());
            }
        }

        //When user clicks on a citation hyperlink
        private void Citation_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Hyperlink link && link.DataContext is Citation c)
            {
                var full = ResolvePath(c.Path);
                TryOpenFile(full);
            }
        }

        //Appends a new line to the log box with a timestamp
        private void Log(string line)
        {
            LogBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {line}\r\n");
            LogBox.ScrollToEnd();
        }

        //Turns any path coming from API into a full local path
        private string ResolvePath(string apiPath)
        {
            if (string.IsNullOrWhiteSpace(apiPath))
                return apiPath;

            var path = apiPath.Replace('/', '\\'); //Normalize slashes for Windows

            //If already an absolute path and file exists, return it as-is
            if (Path.IsPathRooted(path) && File.Exists(path))
                return Path.GetFullPath(path);

            //Check if the user selected a workspace folder
            var workspace = DataFolderBox.Text?.Trim();
            if (!string.IsNullOrEmpty(workspace) && Directory.Exists(workspace))
            {
                //1.Try the file directly under workspace
                var direct = Path.Combine(workspace, path);
                if (File.Exists(direct))
                    return Path.GetFullPath(direct);

                //2.Try relative to workspace parent directory
                var parent = Directory.GetParent(workspace)?.FullName;
                if (!string.IsNullOrEmpty(parent))
                {
                    var fromParent = Path.Combine(parent, path);
                    if (File.Exists(fromParent))
                        return Path.GetFullPath(fromParent);
                }
            }

            //3.Last resort: check from current working directory
            var cwd = Path.Combine(Directory.GetCurrentDirectory(), path);
            if (File.Exists(cwd))
                return Path.GetFullPath(cwd);

            //4.Fallback: return the path unchanged
            return apiPath;
        }

        //Tries to open a file using the default app on the system
        private void TryOpenFile(string fullPath)
        {
            if (File.Exists(fullPath))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = fullPath,
                    UseShellExecute = true
                });
                StatusText.Text = "";
            }
            else
            {
                StatusText.Text = "Path does not exist.";
            }
        }

        //Updates status bar message
        private void Status(string msg) => StatusText.Text = msg;

        //Parses int boxes, falling back to a default if invalid
        private static int SafeInt(string s, int def) => int.TryParse(s, out var v) ? v : def;
    }

    //Represents a single file hit shown in results
    public class FileItem
    {
        public string Path { get; set; }
        public int Page { get; set; }
        public string Snippet { get; set; }
    }

    //Represents a citation item shown under the answer
    public class Citation
    {
        public string Path { get; set; }
        public int Page { get; set; }
        public string ChunkId { get; set; }
    }
}
