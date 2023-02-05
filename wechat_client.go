package main
import (
	"fmt"
	"strings"
	"net/http"
	"io/ioutil"
	"time"
	"os"
	"bytes"
	"encoding/json"
	"github.com/eatmoreapple/openwechat"
)

func Use(vals ...interface{}) {
    for _, val := range vals {
        _ = val
    }
}

type SendTextRequest struct {
	InGroup     bool    `json:"in_group"`   //本来想用于区分在群聊和非群聊时的上下文记忆规则，但是最终没有实现...
	UserID	    string 	`json:"user_id"`
	Text        string  `json:"text"`
}

type SendImageRequest struct {
	UserName	string  `json:"user_name"`
	FileName    string  `json:"filename"`
	HasError    bool    `json:"error"`
	ErrorMessage string `json:"error_msg"`
}

type GenerateImageRequest struct {
	UserName    string 	`json:"user_name"`
	Prompt      string  `json:"prompt"`
}

func HttpPost(url string, data interface{}, timelim int) []byte {
    // 超时时间
	timeout, _ := time.ParseDuration(fmt.Sprintf("%ss", timelim))

    client := &http.Client{Timeout: timeout}
    jsonStr, _ := json.Marshal(data)
    resp, err := client.Post(url, "application/json", bytes.NewBuffer(jsonStr))
    if err != nil {
        return []byte("")
    }
    defer resp.Body.Close()

    result, _ := ioutil.ReadAll(resp.Body)
    return result

// ———————————————
// 版权声明：本文为CSDN博主「gaoluhua」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
// 原文链接：https://blog.csdn.net/gaoluhua/article/details/124855716
}

func main() {

    bot := openwechat.DefaultBot(openwechat.Desktop) // 桌面模式，上面登录不上的可以尝试切换这种模式
	reloadStorage := openwechat.NewJsonFileHotReloadStorage("storage.json")
	defer reloadStorage.Close()

	err := bot.PushLogin(reloadStorage, openwechat.NewRetryLoginOption())
	if err != nil {
		fmt.Println(err)
		return
	}
	
	// 获取登陆的用户
	self, err := bot.GetCurrentUser()
	if err != nil {
		fmt.Println(err)
		return
	}

	Use(self)

	// 注册消息处理函数
	bot.MessageHandler = func(msg *openwechat.Message) {
		if msg.IsTickledMe() {
			msg.ReplyText("别拍了，机器人是会被拍坏掉的。")
			return
		}

		if !msg.IsText() {
			return
		}

	//	fmt.Println(msg.Content)

		content := msg.Content
		if msg.IsSendByGroup() && !msg.IsAt() {
			return
		}

		if msg.IsSendByGroup() && msg.IsAt() {
			atheader := fmt.Sprintf("@%s", self.NickName)
			//fmt.Println(atheader)
			if strings.HasPrefix(content, atheader) {
				content = strings.TrimLeft(content[len(atheader):], "  \t\n")
			}
		}
		//fmt.Println(content)

		if strings.HasPrefix(content, "生成图片") {
			// 调用Stable Diffusion 
			// msg.ReplyText("这个功能还没有实现，可以先期待一下~")
			sender, _ := msg.Sender()

			content = strings.TrimLeft(content[len("生成图片"):], " \t\n")

			resp_raw := HttpPost("http://localhost:11111/draw", GenerateImageRequest{UserName : sender.ID(), Prompt : content}, 120)
			if len(resp_raw) == 0 {
				msg.ReplyText("生成图片出错啦QwQ，或许可以再试一次")
				return
			}

			resp := SendImageRequest{}
			json.Unmarshal(resp_raw, &resp)
			//fmt.Println(resp.FileName)
			if resp.HasError {
				msg.ReplyText( fmt.Sprintf("生成图片出错啦QwQ，错误信息是：%s", resp.ErrorMessage) )
			} else {
				img, _ := os.Open(resp.FileName)
				defer img.Close()
				msg.ReplyImage(img)
			}

		} else {
			// 调用ChatGPT

			sender, _ := msg.Sender()
			//var group openwechat.Group{} = nil
			var group *openwechat.Group = nil

			if msg.IsSendByGroup() {
				group = &openwechat.Group{User : sender}
			}

			if content == "重置上下文" {
				if !msg.IsSendByGroup() {
					HttpPost("http://localhost:11111/chat_clear", SendTextRequest{InGroup : msg.IsSendByGroup(), UserID : sender.ID(), Text : ""}, 60)
				} else {
					HttpPost("http://localhost:11111/chat_clear", SendTextRequest{InGroup : msg.IsSendByGroup(), UserID : group.ID(), Text : ""}, 60)
				}
				msg.ReplyText("OK，我忘掉了之前的上下文。")
				return
			}

			resp := SendTextRequest{}
			resp_raw := []byte("")

			if !msg.IsSendByGroup() {
				resp_raw = HttpPost("http://localhost:11111/chat", SendTextRequest{InGroup : false, UserID : sender.ID(), Text : msg.Content}, 60)
			} else {
				resp_raw = HttpPost("http://localhost:11111/chat", SendTextRequest{InGroup : false, UserID : group.ID(), Text : msg.Content}, 60)
			}
			if len(resp_raw) == 0 {
				msg.ReplyText("运算超时了QAQ，或许可以再试一次。")
				return
			}

			json.Unmarshal(resp_raw, &resp)

			if len(resp.Text) == 0 {
				msg.ReplyText("GPT对此没有什么想说的，换个话题吧。")
			} else {
				if msg.IsSendByGroup() {
					sender_in_group, _ := msg.SenderInGroup()
					nickname := sender_in_group.NickName
					msg.ReplyText(fmt.Sprintf("@%s\n%s\n-------------------\n%s", nickname, content, resp.Text))
				} else {
					msg.ReplyText(resp.Text)
				}
			}

		}
	}
	
	bot.Block()
}